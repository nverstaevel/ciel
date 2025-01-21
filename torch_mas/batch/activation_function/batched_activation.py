import torch

from . import BaseActivation
from ...common.orthotopes.base import (
    batch_intersect_point,
    batch_create_hypercube,
    batch_intersect_hypercubes,
    batch_update_hypercube,
)

batch_update_hypercubes = torch.vmap(batch_update_hypercube, in_dims=(None, 0, None))
batch_batch_intersect_points = torch.vmap(batch_intersect_point)
batch_batch_update_hypercube = torch.vmap(batch_update_hypercube, in_dims=(None, 0, 0))


class BatchedActivation(BaseActivation):

    def neighbors(self, X, side_length, batch_size_orthotopes=1024):
        neighborhood = batch_create_hypercube(
            X, torch.vstack([side_length] * X.size(0))
        )

        neighbor_mask = torch.empty(
            (neighborhood.size(0), self.orthotopes.size(0)),
            dtype=torch.bool,
            device=self.device,
        )

        for i in range(0, self.orthotopes.size(0), batch_size_orthotopes):
            neighbor_mask[:, i : i + batch_size_orthotopes] = (
                batch_intersect_hypercubes(
                    neighborhood, self.orthotopes[i : i + batch_size_orthotopes]
                )
            )

        return neighbor_mask

    def immediate_expandable(self, X, batch_size_orthotopes=1024):
        all_expanded_mask = torch.empty(
            (X.size(0), self.orthotopes.size(0)), dtype=torch.bool, device=self.device
        )

        for i in range(0, self.orthotopes.size(0), batch_size_orthotopes):
            orthotopes_batch = self.orthotopes[i : i + batch_size_orthotopes]

            expanded_neighbors = batch_update_hypercubes(
                orthotopes_batch,
                X,
                torch.full((orthotopes_batch.size(0),), self.alpha, device=self.device),
            )  # (batch_size, n_agents, in_dim, 2) possible shape of each agent for each x
            expanded_mask = batch_batch_intersect_points(
                expanded_neighbors,  # (batch_size*n_agents, in_dim, 2)
                X,  # (batch_size, in_dim)
            )  # (batch_size, n_agents)

            all_expanded_mask[:, i : i + batch_size_orthotopes] = expanded_mask

        return all_expanded_mask

    def update(
        self, X, agents_mask, good, bad, no_activated=False, batch_size_orthotopes=1024
    ):
        batch_size = X.size(0)
        self.goods += good.sum(0).view(self.n_agents, 1)
        self.bads += bad.sum(0).view(self.n_agents, 1)

        for i in range(0, self.orthotopes.size(0), batch_size_orthotopes):

            current_batch_size = self.orthotopes[i : i + batch_size_orthotopes].size(0)

            alphas = torch.zeros(
                (batch_size, current_batch_size), device=self.device
            )  # (batch_size, n_agents)

            current_bad = bad[:, i : i + batch_size_orthotopes]

            alphas = torch.where(
                agents_mask[:, i : i + batch_size_orthotopes]
                & no_activated.view(batch_size, 1)
                & ~current_bad,
                self.alpha,
                alphas,
            )

            alphas = torch.where(
                agents_mask[:, i : i + batch_size_orthotopes]
                & ~no_activated.view(batch_size, 1)
                & current_bad,
                -self.alpha,
                alphas,
            )

            orthotopes_batch = self.orthotopes[i : i + batch_size_orthotopes]
            updated_orthotopes = batch_batch_update_hypercube(
                orthotopes_batch, X, alphas
            )  # (batch_size, n_agents, in_dim, 2)

            deltas = (
                updated_orthotopes - self.orthotopes[i : i + batch_size_orthotopes]
            )  # (batch_size, n_agents, in_dim, 2)
            deltas = deltas.sum(dim=0)
            self.orthotopes[i : i + batch_size_orthotopes] += deltas
