import torch
from .activation_interface import ActivationInterface
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
        agents_mask = batch_intersect_points(self.orthotopes, X)
        return agents_mask

    def neighbors(self, X, side_length):
        neighborhood = batch_create_hypercube(
            X, torch.vstack([side_length] * X.size(0))
        )
        neighbor_mask = batch_intersect_hypercubes(neighborhood, self.orthotopes)
        return neighbor_mask

    def immediate_expandable(self, X):
        expanded_neighbors = batch_update_hypercubes(
            self.orthotopes,
            X,
            torch.full((self.n_agents,), self.alpha, device=self.device),
        )  # (batch_size, n_agents, in_dim, 2) possible shape of each agent for each x
        expanded_mask = batch_batch_intersect_points(
            expanded_neighbors,  # (batch_size*n_agents, in_dim, 2)
            X,  # (batch_size, in_dim)
        )  # (batch_size, n_agents)
        return expanded_mask

    def update(self, X, agents_mask, good, bad, no_activated=False):
        batch_size = X.size(0)
        alphas = torch.zeros(
            (batch_size, self.n_agents), device=self.device
        )  # (batch_size, n_agents)
        alphas = torch.where(
            agents_mask & no_activated.view(batch_size, 1) & ~bad, self.alpha, alphas
        )
        alphas = torch.where(
            agents_mask & ~no_activated.view(batch_size, 1) & bad, -self.alpha, alphas
        )

        updated_orthotopes = batch_batch_update_hypercube(
            self.orthotopes, X, alphas
        )  # (batch_size, n_agents, in_dim, 2)
        deltas = (
            updated_orthotopes - self.orthotopes
        )  # (batch_size, n_agents, in_dim, 2)
        deltas = deltas.sum(dim=0)  # (n_agents, in_dim, 2)
        self.orthotopes += deltas

    def dist_to_border(self, X, agents_mask):
        return batch_dist_points_to_border(self.orthotopes[agents_mask], X)
