import torch
import copy

from . import BaseActivation
from ...common.orthotopes.base import (
    batch_intersect_points,
    batch_intersect_point,
    batch_create_hypercube,
    batch_intersect_hypercubes,
    batch_update_hypercube,
    batch_dist_points_to_border,
)

batch_update_hypercubes = torch.vmap(batch_update_hypercube, in_dims=(None, 0, None))
batch_batch_intersect_points = torch.vmap(batch_intersect_point)
batch_batch_update_hypercube = torch.vmap(batch_update_hypercube, in_dims=(None, 0, 0))


class BaseActivation(BaseActivation):

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

        all_updated_orthotopes = torch.empty(
            (X.size(0), self.orthotopes.size(0), X.size(-1), 2), device=self.device
        )
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

            all_updated_orthotopes[:, i : i + batch_size_orthotopes] = (
                updated_orthotopes
            )

        deltas = (
            all_updated_orthotopes - self.orthotopes
        )  # (batch_size, n_agents, in_dim, 2)
        deltas = deltas.sum(dim=0)  # (n_agents, in_dim, 2)
        self.orthotopes += deltas
