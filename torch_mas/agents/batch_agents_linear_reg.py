import torch
from torch_mas.agents import BatchAgents
from torch_mas.agents.agents_linear_reg import AgentsLinear
from torch_mas.linear_models import (
    batch_fit_linear_regression,
    batch_predict_linear_regression,
)
from torch_mas.hypercubes import *

batch_update_hypercubes = torch.vmap(batch_update_hypercube, in_dims=(None, 0, None))
batch_batch_intersect_points = torch.vmap(batch_intersect_point)
batch_batch_update_hypercube = torch.vmap(batch_update_hypercube, in_dims=(None, 0, 0))


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
    sorted_idxs = torch.argsort(cat_masks, descending=True)

    return new_mem_X[sorted_idxs][:mem_length], new_mem_y[sorted_idxs][:mem_length]


_batch_update_memory = torch.vmap(_update_memory, in_dims=(None, None, 0, 0, 0, 0))


class BatchLinearAgent(AgentsLinear, BatchAgents):
    def __init__(self, input_dim, output_dim, memory_length, alpha, l1=0.1) -> None:
        super().__init__(input_dim, output_dim, memory_length, alpha, l1)

    def create_agents(self, X, agents_to_create, side_lengths):
        """Create agents

        Args:
            X (Tensor): (batch_size, n_dim)
            agents_to_create (BoolTensor): (batch_size,)
            side_lengths (Tensor): (batch_size, n_dim)

        Returns:
            BoolTensor: (n_created, batch_size,)
        """
        batch_size = X.size(0)
        lows = X - side_lengths / 2
        highs = X + side_lengths / 2
        hypercubes = torch.stack([lows, highs], dim=-1)  # (batch_size,)
        agents_mask = (
            batch_intersect_points(hypercubes, X) & agents_to_create
        )  # (batch_size, n_hypercubes)

        # reduce number of created if possible
        covered_mask = torch.where(agents_to_create, False, True)  # (batch_size,)
        selected_mask = torch.zeros(batch_size, dtype=torch.bool)
        while not covered_mask.all():
            new_covered_mask = agents_mask | covered_mask  # (batch_size, n_hypercubes)
            n_newly_covered = torch.sum(
                new_covered_mask ^ covered_mask, dim=-1
            )  # (batch_size,)
            selected_id = torch.argmax(n_newly_covered, dim=-1)
            selected_mask[selected_id] = True
            covered_mask = new_covered_mask[selected_id]

        agents_to_create = selected_mask
        created_idxs = super().create_agents(
            X[agents_to_create], side_lengths[agents_to_create]
        )
        n_created = created_idxs.size(0)
        models_to_init = torch.zeros(
            (n_created, batch_size), dtype=torch.bool
        )  # (n_created, batch_size)
        models_to_init = agents_mask[agents_to_create]
        return models_to_init

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

    def _update_memories(
        self, X: torch.Tensor, y: torch.Tensor, agent_mask: torch.BoolTensor
    ):
        mem_masks = (
            torch.arange(self.memory_length) < self.memory_sizes
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
        agents_to_update = self._update_memories(X, y, agent_mask)

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
