import torch

from . import BaseActivation
from ...common.orthotopes.base import (
    batch_intersect_point,
    batch_create_hypercube,
    batch_intersect_hypercubes,
    batch_update_hypercube,
    batch_dist_points_to_border,
)

batch_update_hypercubes = torch.vmap(batch_update_hypercube, in_dims=(None, 0, None))
batch_batch_intersect_points = torch.vmap(batch_intersect_point)
batch_batch_update_hypercube = torch.vmap(batch_update_hypercube, in_dims=(None, 0, 0))


class BatchedActivation(BaseActivation):
    def __init__(
        self, input_dim, output_dim, alpha, device="cpu", batch_fnc=1024, **kwargs
    ):
        super().__init__(input_dim, output_dim, alpha, device=device, **kwargs)
        self.batch_fnc = batch_fnc

    def neighbors(self, X, side_length):
        neighborhood = batch_create_hypercube(
            X, torch.vstack([side_length] * X.size(0))
        )

        neighbor_mask = torch.empty(
            (neighborhood.size(0), self.orthotopes.size(0)),
            dtype=torch.bool,
            device=self.device,
        )

        for i in range(0, self.orthotopes.size(0), self.batch_fnc):
            neighbor_mask[:, i : i + self.batch_fnc] = batch_intersect_hypercubes(
                neighborhood, self.orthotopes[i : i + self.batch_fnc]
            )

        return neighbor_mask

    def immediate_expandable(self, X):
        all_expanded_mask = torch.empty(
            (X.size(0), self.orthotopes.size(0)), dtype=torch.bool, device=self.device
        )

        for i in range(0, self.orthotopes.size(0), self.batch_fnc):
            orthotopes_batch = self.orthotopes[i : i + self.batch_fnc]

            expanded_neighbors = batch_update_hypercubes(
                orthotopes_batch,
                X,
                torch.full((orthotopes_batch.size(0),), self.alpha, device=self.device),
            )  # (batch_size, n_agents, in_dim, 2) possible shape of each agent for each x
            expanded_mask = batch_batch_intersect_points(
                expanded_neighbors,  # (batch_size*n_agents, in_dim, 2)
                X,  # (batch_size, in_dim)
            )  # (batch_size, n_agents)

            all_expanded_mask[:, i : i + self.batch_fnc] = expanded_mask

        return all_expanded_mask

    def update(self, X, agents_mask, good, bad, no_activated=False):
        batch_size = X.size(0)
        self.goods += good.sum(0).view(self.n_agents, 1)
        self.bads += bad.sum(0).view(self.n_agents, 1)

        for i in range(0, self.orthotopes.size(0), self.batch_fnc):
            current_batch_size = self.orthotopes[i : i + self.batch_fnc].size(0)

            alphas = torch.zeros(
                (batch_size, current_batch_size), device=self.device
            )  # (batch_size, n_agents)

            current_bad = bad[:, i : i + self.batch_fnc]

            alphas = torch.where(
                agents_mask[:, i : i + self.batch_fnc]
                & no_activated.view(batch_size, 1)
                & ~current_bad,
                self.alpha,
                alphas,
            )

            alphas = torch.where(
                agents_mask[:, i : i + self.batch_fnc]
                & ~no_activated.view(batch_size, 1)
                & current_bad,
                -self.alpha,
                alphas,
            )

            orthotopes_batch = self.orthotopes[i : i + self.batch_fnc]
            updated_orthotopes = batch_batch_update_hypercube(
                orthotopes_batch, X, alphas
            )  # (batch_size, n_agents, in_dim, 2)

            deltas = (
                updated_orthotopes - self.orthotopes[i : i + self.batch_fnc]
            )  # (batch_size, n_agents, in_dim, 2)
            deltas = deltas.sum(dim=0)
            self.orthotopes[i : i + self.batch_fnc] += deltas

    def dist_to_border(self, X, agents_mask):
        dists_to_border = torch.empty(
            (X.size(0), self.orthotopes[agents_mask].size(0)), device=self.device
        )

        for i in range(0, self.orthotopes[agents_mask].size(0), self.batch_fnc):
            dists_to_border[:, i : i + self.batch_fnc] = batch_dist_points_to_border(
                self.orthotopes[agents_mask][i : i + self.batch_fnc], X
            )

        return dists_to_border
