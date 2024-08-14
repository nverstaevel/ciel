import torch
from torch_mas.hypercubes import *
from abc import ABC, abstractmethod


class Agents(ABC):
    def __init__(self, input_dim, output_dim, memory_length, alpha, l1=0.1) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memory_length = memory_length
        self.alpha = alpha
        self.l1_penalty = l1

        self.hypercubes: torch.Tensor = torch.empty(
            0, input_dim, 2, requires_grad=True
        )  # (n_agents, input_dim, 2) Tensor of hypercubes
        self.feature_memories: torch.Tensor = torch.empty(
            0, memory_length, input_dim, dtype=torch.float
        )  # (n_agents, memory_length,) Tensor of features
        self.target_memories: torch.Tensor = torch.empty(
            0, memory_length, output_dim, dtype=torch.float
        )  # (n_agents, memory_length,) Tensor of targets
        self.memory_sizes: torch.Tensor = torch.empty(
            0, 1, dtype=torch.long
        )  # (n_agents, 1) Tensor of fill memory levels
        self.memory_ptr: torch.Tensor = torch.empty(
            0, 1, dtype=torch.long
        )  # (n_agents, 1) Tensor of fill memory levels

    @property
    def n_agents(self):
        return self.hypercubes.size(0)

    def activated(self, X):
        """Get activated agents mask

        Args:
            X (Tensor): (input_dim,)

        Returns:
            BoolTensor: (n_agents,)
        """
        agent_mask = batch_intersect_point(self.hypercubes, X)
        return agent_mask

    def neighbors(self, X, side_length):
        """Get neighbors agents mask

        Args:
            X (Tensor): (1, input_dim)
            side_length (Tensor): (n_dim,) | (1,)

        Returns:
            BoolTensor: (n_agents,)
        """
        neighborhood = create_hypercube(X.squeeze(0), side_length)
        neighbor_mask = batch_intersect_hypercube(neighborhood, self.hypercubes)
        return neighbor_mask

    def immediate_expandable(self, X, agents_idxs):
        """Get a mask of agents that can do a one-step expansion to include X

        Args:
            X (Tensor): (1, input_dim)
            agents_idxs (BoolTensor): (n_agents,)

        Returns:
            BoolTensor: (n_agents,)
        """
        n_agents = torch.count_nonzero(agents_idxs)
        expanded_neighbors = batch_update_hypercube(
            self.hypercubes[agents_idxs],
            X.squeeze(0),
            torch.full((n_agents,), self.alpha),
        )
        expanded_mask = batch_intersect_point(expanded_neighbors, X)
        return expanded_mask

    def maturity(self, agents_idxs):
        """Get maturity of specified agents

        Args:
            agents_idxs (LongTensor | BoolTensor): (n_agents,)

        Returns:
            BoolTensor: (n_agents, 1)
        """
        return self.memory_sizes[agents_idxs] > (self.input_dim + 1)

    @abstractmethod
    def create_agents(self, X, side_lengths):
        """Create agents

        Args:
            X_idxs (Tensor): (batch_size, n_dim)
            side_lengths (Tensor): (batch_size, n_dim)

        Returns:
            LongTensor: (batch_size,)
        """
        pass

    @abstractmethod
    def destroy_agents(self, idxs):
        """Destroy agents

        Args:
            idxs (LongTensor | BoolTensor): (batch_size,)
        """
        pass

    @abstractmethod
    def update_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        agents_idxs: torch.LongTensor | torch.BoolTensor,
    ):
        """Update the local model of specified agents.

        Args:
            X (Tensor): (1, input_dim)
            y (Tensor): (1, output_dim)
            agents_idxs (LongTensor | BoolTensor): (n_agents,)
        """
        pass

    @abstractmethod
    def update_hypercube(
        self,
        X: torch.Tensor,
        agents_idxs: torch.LongTensor | torch.BoolTensor,
        good: torch.BoolTensor,
        bad: torch.BoolTensor,
        no_activated: bool = False,
    ):
        """Update hypercube of sepcified agents.

        Args:
            X (Tensor): (1, input_dim,)
            agents_idxs (LongTensor | BoolTensor): (n_agents_to_update,) | (n_agents,)
            good (BoolTensor): (n_agents_to_update,) | (n_agents,)
            bad (BoolTensor): (n_agents_to_update,) | (n_agents,)
            no_activated (bool): True if at least 1 agent activated by X
        """
        pass

    @abstractmethod
    def predict(
        self, X: torch.FloatTensor, agents_idxs: torch.LongTensor | torch.BoolTensor
    ):
        """Retrieve predictions from agents

        Args:
            X (FloatTensor): (batch_size, input_dim)
            agents_idxs (LongTensor | BoolTensor): (n_agents,)

        Returns:
            Tensor: (n_agents, batch_size, output_dim)
        """
        pass

    @abstractmethod
    def __call__(
        self, X: torch.FloatTensor, neighborhood_sides: torch.FloatTensor
    ) -> torch.Any:
        pass


# TODO: Give better names to those functions
batch_update_hypercubes = torch.vmap(batch_update_hypercube, in_dims=(None, 0, None))
batch_batch_intersect_points = torch.vmap(batch_intersect_point)
batch_batch_update_hypercube = torch.vmap(batch_update_hypercube, in_dims=(None, 0, 0))


class BatchAgents(Agents):
    def activated(self, X):
        """Get activated agents mask

        Args:
            X (Tensor): (batch_size, input_dim)

        Returns:
            BoolTensor: (batch_size, n_agents)
        """
        agents_mask = batch_intersect_points(self.hypercubes, X)
        return agents_mask

    def neighbors(self, X, side_length):
        """Get neighbors agents mask

        Args:
            X (Tensor): (batch_size, input_dim)
            side_length (Tensor): (n_dim,) | (1,)

        Returns:
            BoolTensor: (batch_size, n_agents)
        """
        neighborhood = batch_create_hypercube(
            X, torch.vstack([side_length] * X.size(0))
        )
        neighbor_mask = batch_intersect_hypercubes(neighborhood, self.hypercubes)
        return neighbor_mask

    def immediate_expandable(self, X):
        """Get a mask of agents that can do a one-step expansion to include X

        Args:
            X (Tensor): (batch_size, input_dim)

        Returns:
            BoolTensor: (batch_size, n_agents)
        """
        expanded_neighbors = batch_update_hypercubes(
            self.hypercubes,
            X,
            torch.full((self.n_agents,), self.alpha),
        )  # (batch_size, n_agents, in_dim, 2) possible shape of each agent for each x
        expanded_mask = batch_batch_intersect_points(
            expanded_neighbors,  # (batch_size*n_agents, in_dim, 2)
            X,  # (batch_size, in_dim)
        )  # (batch_size, n_agents)
        return expanded_mask

    @abstractmethod
    def create_agents(self, X, agents_to_create, side_lengths):
        """Create agents

        Args:
            X (Tensor): (batch_size, n_dim)
            agents_to_create (BoolTensor): (batch_size,)
            side_lengths (Tensor): (batch_size, n_dim)

        Returns:
            BoolTensor: (n_created, batch_size,)
        """
        pass

    @abstractmethod
    def update_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        agent_mask: torch.BoolTensor,
    ):
        """Update the local model of specified agents.

        Args:
            X (Tensor): (batch_size, input_dim)
            y (Tensor): (batch_size, output_dim)
            agent_mask (BoolTensor): (n_agents, batch_size)
        """
        pass

    @abstractmethod
    def update_hypercube(
        self,
        X: torch.Tensor,
        agents_mask: torch.BoolTensor,
        good: torch.BoolTensor,
        bad: torch.BoolTensor,
        no_activated: torch.BoolTensor,
    ):
        """Update hypercube of specified agents.

        Args:
            X (Tensor): (batch_size, input_dim,)
            agents_mask (BoolTensor): (batch_size, n_agents)
            good (BoolTensor): (batch_size, n_agents)
            bad (BoolTensor): (batch_size, n_agents)
            no_activated (BoolTensor): (batch_size,) True if at least 1 agent activated by X
        """
