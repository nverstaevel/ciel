import torch
from torch_mas.agents.agents_linear_reg import AgentsLinear
from torch_mas.linear_models import (
    batch_fit_linear_regression,
    batch_predict_linear_regression,
)
from torch_mas.hypercubes import *

batch_update_hypercubes = torch.vmap(batch_update_hypercube, in_dims=(None, 0, None))
batch_batch_intersect_points = torch.vmap(batch_intersect_point)
batch_batch_update_hypercube = torch.vmap(batch_update_hypercube, in_dims=(None, 0, 0))


class BatchLinearAgent(AgentsLinear):
    def __init__(self, input_dim, output_dim, memory_length, alpha, l1=0.1) -> None:
        super().__init__(input_dim, output_dim, memory_length, alpha, l1)

    def create_agents(self, X, side_lengths):
        """Create agents

        Args:
            X_idxs (Tensor): (batch_size, n_dim)
            side_lengths (Tensor): (batch_size, n_dim)

        Returns:
            BoolTensor: (batch_size,)
        """
        created_idxs = super().create_agents(X, side_lengths)
        created_mask = torch.zeros(self.n_agents, dtype=torch.bool)
        created_mask[created_idxs] = True
        return created_mask

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
        batch_size = X.size(0)
        alphas = torch.zeros((batch_size, self.n_agents))  # (batch_size, n_agents)
        alphas = torch.where(
            agents_mask & no_activated.view(batch_size, 1) & ~bad, self.alpha, alphas
        )
        alphas = torch.where(
            agents_mask & ~no_activated.view(batch_size, 1) & bad, -self.alpha, alphas
        )

        updated_hypercubes = batch_batch_update_hypercube(
            self.hypercubes, X, alphas
        )  # (batch_size, n_agents, in_dim, 2)
        deltas = (
            updated_hypercubes - self.hypercubes
        )  # (batch_size, n_agents, in_dim, 2)
        deltas = deltas.sum(dim=0)  # (n_agents, in_dim, 2)
        self.hypercubes += deltas

    def update_model(
        self, X: torch.Tensor, y: torch.Tensor, agent_mask: torch.BoolTensor
    ):
        """Update the local model of specified agents.

        Args:
            X (Tensor): (batch_size, input_dim)
            y (Tensor): (batch_size, output_dim)
            agent_mask (BoolTensor): (n_agents, batch_size)
        """
        # update memory
        # TODO: vectorize this part of the code to enable fast GPU acceleration
        nb_to_add_per_agent = agent_mask.sum(-1)
        for agent_id, mask in enumerate(agent_mask):
            nb_to_add = nb_to_add_per_agent[agent_id]
            idxs_to_add = (
                torch.arange(nb_to_add) + self.memory_ptr[agent_id]
            ) % self.memory_length
            self.feature_memories[agent_id, idxs_to_add] = X[mask]
            self.target_memories[agent_id, idxs_to_add] = y[mask]

        self.memory_sizes += torch.where(
            self.memory_sizes < self.memory_length, nb_to_add_per_agent.view(-1, 1), 0
        )
        self.memory_ptr += nb_to_add_per_agent.view(-1, 1)

        # update model
        agents_to_update = nb_to_add_per_agent > 0
        # build weights for regression
        weights = (
            torch.arange(self.memory_length) < (self.memory_sizes[agents_to_update])
        ).float()
        # extract memories
        X = self.feature_memories[agents_to_update]
        y = self.target_memories[agents_to_update]
        # update agents
        updated_models = batch_fit_linear_regression(
            X, y, weights, l1_penalty=self.l1_penalty
        )
        # save updated models
        self.models[agents_to_update] = updated_models
