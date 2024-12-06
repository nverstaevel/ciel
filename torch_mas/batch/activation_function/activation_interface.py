import torch
from abc import ABC, abstractmethod


class ActivationInterface(ABC):
    @property
    @abstractmethod
    def n_agents(self):
        raise NotImplementedError()

    @abstractmethod
    def create(
        self,
        X: torch.Tensor,
        side_lengths: torch.Tensor,
    ):
        """Create agents

        Args:
            X (Tensor): (batch_size, n_dim)
            side_lengths (Tensor): (batch_size, n_dim)
        """
        raise NotImplementedError()

    @abstractmethod
    def activated(self, X: torch.Tensor):
        """Get activated agents mask

        Args:
            X (Tensor): (batch_size, input_dim)

        Returns:
            BoolTensor: (batch_size, n_agents)
        """
        raise NotImplementedError()

    @abstractmethod
    def neighbors(self, X: torch.Tensor, side_length: torch.Tensor):
        """Get neighbors agents mask

        Args:
            X (Tensor): (batch_size, input_dim)
            side_length (Tensor): (n_dim,) | (1,)

        Returns:
            BoolTensor: (batch_size, n_agents)
        """
        raise NotImplementedError()

    @abstractmethod
    def immediate_expandable(self, X: torch.Tensor):
        """Get a mask of agents that can do a one-step expansion to include X

        Args:
            X (Tensor): (batch_size, input_dim)

        Returns:
            BoolTensor: (batch_size, n_agents)
        """
        raise NotImplementedError()

    @abstractmethod
    def update(
        self,
        X: torch.Tensor,
        agents_mask: torch.BoolTensor,
        good: torch.BoolTensor,
        bad: torch.BoolTensor,
        no_activated: bool = False,
    ):
        """Update orthotopes of sepcified agents.

        Args:
            X (Tensor): (batch_size, input_dim)
            agents_mask (BoolTensor): (n_agents,)
            good (BoolTensor): (n_agents,)
            bad (BoolTensor): (n_agents,)
            no_activated (bool): True if at least 1 agent activated by X
        """
        raise NotImplementedError()

    @abstractmethod
    def dist_to_border(self, X: torch.Tensor, agents_mask: torch.BoolTensor):
        """_summary_

        Args:
            X (torch.Tensor): (batch_size, input_dim)
            agents_mask (torch.BoolTensor): (n_agents,)

        Returns:
            FloatTensor: (batch_size, n_agents)
        """
        raise NotImplementedError()
