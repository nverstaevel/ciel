import torch
import copy
from .model_interface import InternalModelInterface


class NClass(InternalModelInterface):
    def __init__(self, input_dim, output_dim, memory_length, device="cpu", **kwargs):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memory_length = memory_length

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

        self.base_prediction: torch.Tensor = torch.zeros(
            0, dtype=torch.float, requires_grad=False, device=device
        )  # (n_agents) Tensor of base prediction

        self.initialized: torch.Tensor = torch.zeros(
            0, dtype=torch.bool, requires_grad=False, device=device
        )  # (n_agents) Tensor of base prediction

    @property
    def n_agents(self):
        return self.models.size(0)

    def create(self, X):
        batch_size = X.size(0)
        base_prediction = torch.zeros(
            (batch_size,), dtype=torch.float, device=self.device
        )
        initialized = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
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

        self.base_prediction = torch.vstack(
            [self.base_prediction.unsqueeze(-1), base_prediction.unsqueeze(-1)]
        ).squeeze(-1)
        self.initialized = torch.cat([self.initialized, initialized])
        self.feature_memories = torch.vstack([self.feature_memories, feature_memories])
        self.target_memories = torch.vstack([self.target_memories, target_memories])
        self.memory_sizes = torch.vstack([self.memory_sizes, memory_size])
        self.memory_ptr = torch.vstack([self.memory_ptr, memory_ptr])

    def maturity(self, agents_mask):
        return self.memory_sizes[agents_mask] > (self.input_dim + 1)

    def update(self, X, y, agent_mask):
        index_mask = torch.zeros_like(self.initialized)
        index_mask[agent_mask] = True

        mask = index_mask & ~self.initialized
        self.base_prediction[mask] = y

        self.initialized[agent_mask] = True

    def __call__(self, X, agents_mask=None):
        return (
            self.base_prediction[agents_mask]
            .unsqueeze(1)
            .unsqueeze(2)
            .repeat(1, X.size(0), 1)
        )

    def clone(self):
        cloned_self = copy.copy(self)  # shallow copy
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(cloned_self, attr_name, attr_value.clone())
        return cloned_self
