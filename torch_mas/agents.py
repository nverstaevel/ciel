import torch
from torch_mas.linear_models import (
    batch_fit_linear_regression,
    batch_predict_linear_regression,
)
from torch_mas.hypercubes import *


class Agents:
    def __init__(self, input_dim, output_dim, memory_length, l1=0.1) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memory_length = memory_length
        self.l1_penalty = l1

        self.hypercubes: torch.Tensor = torch.empty(
            0, input_dim, 2, requires_grad=True
        )  # (n_agents, input_dim, 2) Tensor of hypercubes
        self.models: torch.Tensor = torch.empty(
            0, input_dim + 1, output_dim, dtype=torch.float, requires_grad=False
        )  # (n_agents, input_dim+1, output_dim) Tensor of linear models
        self.feature_memories: torch.Tensor = torch.empty(
            0, memory_length, input_dim, dtype=torch.float
        )  # (n_agents, memory_length,) Tensor of features
        self.target_memories: torch.Tensor = torch.empty(
            0, memory_length, output_dim, dtype=torch.float
        )  # (n_agents, memory_length,) Tensor of targets
        self.memory_sizes: torch.Tensor = torch.empty(
            0, 1, dtype=torch.long
        )  # (n_agents, 1) Tensor of fill memory levels

    @property
    def n_agents(self):
        return self.hypercubes.size(0)

    def create_agents(self, X, side_lengths):
        """Create agents

        Args:
            X_idxs (Tensor): (batch_size, n_dim)
            side_lengths (Tensor): (batch_size, n_dim)

        Returns:
            LongTensor: (batch_size,)
        """
        batch_size = X.size(0)
        lows = X - side_lengths / 2
        highs = X + side_lengths / 2

        hypercubes = torch.stack([lows, highs], dim=-1)
        models = torch.zeros((batch_size, self.input_dim + 1, self.output_dim))
        feature_memories = torch.zeros(
            (batch_size, self.memory_length, self.input_dim), dtype=torch.float
        )
        target_memories = torch.zeros(
            (batch_size, self.memory_length, self.output_dim), dtype=torch.float
        )
        memory_size = torch.zeros((batch_size, 1), dtype=torch.long)

        created_idxs = torch.arange(0, batch_size) + self.hypercubes.size(0)

        self.hypercubes = torch.vstack([self.hypercubes, hypercubes])
        self.models = torch.vstack([self.models, models])
        self.feature_memories = torch.vstack([self.feature_memories, feature_memories])
        self.target_memories = torch.vstack([self.target_memories, target_memories])
        self.memory_sizes = torch.vstack([self.memory_sizes, memory_size])
        return created_idxs.long()

    def destroy_agents(self, idxs):
        """Destroy agents

        Args:
            idxs (LongTensor | BoolTensor): (batch_size,)
        """
        mask = torch.ones(self.hypercubes.shape[0], dtype=torch.bool)
        mask[idxs] = 0
        self.hypercubes = self.hypercubes[mask]
        self.feature_memories = self.feature_memories[mask]
        self.target_memories = self.target_memories[mask]
        self.models = self.models[mask]
        self.memory_sizes = self.memory_sizes[mask]

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
        # add X_idx indexes to memories
        idxs_to_add = (self.memory_sizes[agents_idxs] % self.memory_length).squeeze(-1)
        self.feature_memories[agents_idxs, idxs_to_add] = X
        self.target_memories[agents_idxs, idxs_to_add] = y
        self.memory_sizes[agents_idxs] += 1
        # build weights for regression
        weights = (
            torch.arange(self.memory_length) < (self.memory_sizes[agents_idxs])
        ).float()
        # extract memories
        X = self.feature_memories[agents_idxs]
        y = self.target_memories[agents_idxs]
        # update agents
        updated_models = batch_fit_linear_regression(
            X, y, weights, l1_penalty=self.l1_penalty
        )
        # save updated models
        self.models[agents_idxs] = updated_models

    def update_hypercube(
        self,
        X: torch.Tensor,
        agents_idxs: torch.LongTensor | torch.BoolTensor,
        alphas: torch.FloatTensor,
    ):
        """Update hypercube of sepcified agents.

        Args:
            X (Tensor): (1, input_dim,)
            agents_idxs (LongTensor | BoolTensor): (n_agents,)
            alphas (FloatTensor): (n_agents,)
        """
        updated_hypercubes = batch_update_hypercube(
            self.hypercubes[agents_idxs], X.squeeze(0), alphas
        )
        self.hypercubes[agents_idxs] = updated_hypercubes

    def activated(self, X):
        """Get activated agents mask

        Args:
            X (Tensor): (input_dim,)

        Returns:
            BoolTensor: (n_agents_activated,)
        """
        agent_mask = batch_intersect_point(self.hypercubes, X)
        return agent_mask

    def neighbors(self, X, side_length):
        """Get neighbors agents mask

        Args:
            X (Tensor): (1, input_dim)
            side_length (Tensor): (n_dim,) | (1,)

        Returns:
            BoolTensor: (n_agents_neighbors,)
        """
        neighborhood = create_hypercube(X.squeeze(0), side_length)
        neighbor_mask = batch_intersect_hypercube(neighborhood, self.hypercubes)
        return neighbor_mask

    def maturity(self, agents_idxs):
        """Get maturity of specified agents

        Args:
            agents_idxs (LongTensor | BoolTensor): (n_agents,)

        Returns:
            BoolTensor: (n_agents,)
        """
        # return self.memory_sizes[agents_idxs] > (self.input_dim+self.output_dim)
        return self.memory_sizes[agents_idxs] > (self.input_dim + 1)

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
        y_pred = batch_predict_linear_regression(X, self.models[agents_idxs])
        return y_pred
