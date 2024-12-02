import torch
from .model_interface import InternalModelInterface
from ...common.models.linear_models import (
    batch_fit_linear_regression,
    batch_predict_linear_regression,
)


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

    def update(self, X, y, agent_mask):
        # add X_idx indexes to memories
        idxs_to_add = (self.memory_ptr[agent_mask] % self.memory_length).squeeze(-1)
        self.feature_memories[agent_mask, idxs_to_add] = X
        self.target_memories[agent_mask, idxs_to_add] = y
        self.memory_sizes[agent_mask] += torch.where(
            self.memory_sizes[agent_mask] < self.memory_length, 1, 0
        )
        self.memory_ptr[agent_mask] += 1
        # build weights for regression
        weights = (
            torch.arange(self.memory_length) < (self.memory_sizes[agent_mask])
        ).float()
        # extract memories
        X = self.feature_memories[agent_mask]
        y = self.target_memories[agent_mask]
        # update agents
        updated_models = batch_fit_linear_regression(X, y, weights, l1_penalty=self.l1)
        # save updated models
        self.models[agent_mask] = updated_models

    def __call__(self, X, agents_mask=None):
        return batch_predict_linear_regression(X, self.models[agents_mask])
