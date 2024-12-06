import torch
from abc import ABC, abstractmethod


class InternalModelInterface(ABC):
    @property
    @abstractmethod
    def n_agents(self):
        raise NotImplementedError()

    @abstractmethod
    def create(self, X: torch.Tensor):
        """Create new internal models

        Args:
            X (Tensor): (batch_size, n_dim)
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, X: torch.Tensor, y: torch.Tensor, agent_mask: torch.BoolTensor):
        """Update the local model of specified agents.

        Args:
            X (Tensor): (1, input_dim)
            y (Tensor): (1, output_dim)
            agent_mask (BoolTensor): (n_agents,)
        """
        raise not NotImplementedError()

    @abstractmethod
    def maturity(self, agents_mask: torch.BoolTensor):
        """Get maturity of specified agents

        Args:
            agents_mask (BoolTensor): (n_agents,)

        Returns:
            BoolTensor: (n_agents, 1)
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, X: torch.Tensor, agents_mask: torch.BoolTensor = None):
        """Retrieve predictions from agents

        Args:
            X (FloatTensor): (batch_size, input_dim)
            agents_mask (BoolTensor): (n_agents,)

        Returns:
            Tensor: (n_agents, batch_size, output_dim)
        """
        raise NotImplementedError()

    @abstractmethod
    def clone(self):
        """Generates a deepcopy of the current state of the activation function."""
        raise NotImplementedError()
