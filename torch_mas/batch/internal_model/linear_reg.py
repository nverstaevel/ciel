import torch
import copy
from .model_interface import InternalModelInterface
from ...common.models.linear_models import (
    batch_fit_linear_regression,
    batch_predict_linear_regression,
)


def _update_memory(X, y, mem_X, mem_y, mem_mask, batch_mask):
    """_summary_

    Args:
        X (Tensor): (batch_size, in_dim)
        y (Tensor): (batch_size, out_dim)
        mem_X (Tensor): (memory_length, in_dim)
        mem_y (Tensor): (memory_length, out_dim)
        mem_mask (Tensor): (memory_length,)
        agent_mask (Tensor): (batch_size,)
    """
    mem_length = mem_X.size(0)
    cat_masks = torch.cat([batch_mask, mem_mask])
    new_mem_X = torch.cat([X, mem_X])
    new_mem_y = torch.cat([y, mem_y])
    sorted_idxs = torch.argsort(cat_masks.float(), descending=True)

    return new_mem_X[sorted_idxs][:mem_length], new_mem_y[sorted_idxs][:mem_length]


_batch_update_memory = torch.vmap(_update_memory, in_dims=(None, None, 0, 0, 0, 0))


class LinearWithMemory(InternalModelInterface):
    def __init__(
        self, input_dim, output_dim, memory_length, l1, device="cpu", **kwargs
    ):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memory_length = memory_length
        self.l1 = l1

        self.feature_memories: torch.Tensor = torch.empty(
            0, memory_length, input_dim, dtype=torch.float, device=device
        )  # (n_agents, memory_length,) Tensor of features
        self.target_memories: torch.Tensor = torch.empty(
            0, memory_length, output_dim, dtype=torch.float, device=device
        )  # (n_agents, memory_length,) Tensor of targets
        self.memory_sizes: torch.Tensor = torch.empty(
            0, 1, dtype=torch.long, device=device
        )  # (n_agents, 1) Tensor of fill memory levels
        self.memory_ptr: torch.Tensor = torch.empty(
            0, 1, dtype=torch.long, device=device
        )  # (n_agents, 1) Tensor of fill memory levels
        self.models: torch.Tensor = torch.empty(
            0,
            input_dim + 1,
            output_dim,
            dtype=torch.float,
            requires_grad=False,
            device=device,
        )  # (n_agents, input_dim+1, output_dim) Tensor of linear models

    @property
    def n_agents(self):
        return self.models.size(0)

    def destroy(self, agents_mask):
        self.feature_memories = self.feature_memories[~agents_mask]
        self.target_memories = self.target_memories[~agents_mask]
        self.memory_sizes = self.memory_sizes[~agents_mask]
        self.memory_ptr = self.memory_ptr[~agents_mask]
        self.models = self.models[~agents_mask]

    def create(self, X):
        batch_size = X.size(0)
        models = torch.zeros(
            (batch_size, self.input_dim + 1, self.output_dim), device=self.device
        )
        feature_memories = torch.zeros(
            (batch_size, self.memory_length, self.input_dim),
            dtype=torch.float,
            device=self.device,
        )
        target_memories = torch.zeros(
            (batch_size, self.memory_length, self.output_dim),
            dtype=torch.float,
            device=self.device,
        )
        memory_size = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        memory_ptr = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)

        self.models = torch.vstack([self.models, models])
        self.feature_memories = torch.vstack([self.feature_memories, feature_memories])
        self.target_memories = torch.vstack([self.target_memories, target_memories])
        self.memory_sizes = torch.vstack([self.memory_sizes, memory_size])
        self.memory_ptr = torch.vstack([self.memory_ptr, memory_ptr])

    def maturity(self, agents_mask):
        return self.memory_sizes[agents_mask] > (self.input_dim + 1)

    def _update_memories(
        self, X: torch.Tensor, y: torch.Tensor, agent_mask: torch.BoolTensor
    ):
        mem_masks = (
            torch.arange(self.memory_length, device=self.device) < self.memory_sizes
        )  # (n_agents, memory_length)
        nb_to_add_per_agent = agent_mask.sum(-1)
        agents_to_update = nb_to_add_per_agent > 0
        self.memory_sizes = torch.clip(
            self.memory_sizes + nb_to_add_per_agent.view(-1, 1), max=self.memory_length
        )

        (
            self.feature_memories[agents_to_update],
            self.target_memories[agents_to_update],
        ) = _batch_update_memory(
            X,
            y,
            self.feature_memories[agents_to_update],
            self.target_memories[agents_to_update],
            mem_masks[agents_to_update],
            agent_mask[agents_to_update],
        )

        return agents_to_update

    def update(self, X, y, agent_mask):
        # update memory
        agents_to_update = self._update_memories(X, y, agent_mask)

        # build weights for regression
        weights = (
            torch.arange(self.memory_length, device=self.device)
            < (self.memory_sizes[agents_to_update])
        ).float()
        # extract memories
        X = self.feature_memories[agents_to_update]
        y = self.target_memories[agents_to_update]
        # update agents
        updated_models = batch_fit_linear_regression(X, y, weights, l1_penalty=self.l1)
        # save updated models
        self.models[agents_to_update] = updated_models

    def __call__(self, X, agents_mask=None):
        return batch_predict_linear_regression(X, self.models[agents_mask])

    def clone(self):
        cloned_self = copy.copy(self)  # shallow copy
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(cloned_self, attr_name, attr_value.clone())
        return cloned_self
