import torch
from .activation_interface import ActivationInterface
from ...common.orthotopes.base import (
    batch_create_hypercube,
    batch_intersect_hypercubes,
    batch_intersect_point,
    batch_intersect_hypercube,
    batch_update_hypercube,
    batch_dist_points_to_border,
    create_hypercube,
)

batch_update_hypercubes = torch.vmap(batch_update_hypercube, in_dims=(None, 0, None))
batch_batch_intersect_points = torch.vmap(batch_intersect_point)
batch_batch_update_hypercube = torch.vmap(batch_update_hypercube, in_dims=(None, 0, 0))


class BaseActivation(ActivationInterface):
    def __init__(self, input_dim, output_dim, alpha, device="cpu", **kwargs):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = torch.tensor(alpha, device=self.device)

        self.orthotopes: torch.Tensor = torch.empty(
            0, input_dim, 2, device=device
        )  # (n_agents, input_dim, 2) Tensor of orthotopes

    @property
    def n_agents(self):
        return self.orthotopes.size(0)

    def create(self, X, side_lengths):
        lows = X - side_lengths / 2
        highs = X + side_lengths / 2

        orthotopes = torch.stack([lows, highs], dim=-1)
        self.orthotopes = torch.vstack([self.orthotopes, orthotopes])

    def activated(self, X):
        agents_mask = batch_intersect_point(self.orthotopes, X)
        return agents_mask

    def neighbors(self, X, side_length):
        if X.size(0) > 1:
            neighborhood = batch_create_hypercube(
                X, torch.vstack([side_length] * X.size(0))
            )
            neighbor_mask = batch_intersect_hypercubes(neighborhood, self.orthotopes)
        else:
            neighborhood = create_hypercube(X.squeeze(0), side_length)
            neighbor_mask = batch_intersect_hypercube(neighborhood, self.orthotopes)
        return neighbor_mask

    def immediate_expandable(self, X, agents_mask):
        n_agents = torch.count_nonzero(agents_mask)
        expanded_neighbors = batch_update_hypercube(
            self.orthotopes[agents_mask],
            X.squeeze(0),
            torch.full((n_agents,), self.alpha),
        )
        expanded_mask = batch_intersect_point(expanded_neighbors, X)
        return expanded_mask

    def update(self, X, agents_mask, good, bad, no_activated=False):
        n_agents = (
            agents_mask.size(0)
            if isinstance(agents_mask, torch.LongTensor)
            else agents_mask.count_nonzero()
        )
        alphas = torch.zeros((n_agents, 1))
        if no_activated:
            alphas[~bad] = self.alpha
        else:
            alphas[bad] = -self.alpha

        updated_orthotopes = batch_update_hypercube(
            self.orthotopes[agents_mask], X.squeeze(0), alphas
        )
        self.orthotopes[agents_mask] = updated_orthotopes

    def dist_to_border(self, X, agents_mask):
        return batch_dist_points_to_border(self.orthotopes[agents_mask], X)
