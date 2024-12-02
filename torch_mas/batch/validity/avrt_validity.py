import torch

from . import BaseValidity
from ...common.orthotopes.avrt import (
    batch_init_deltas_avrt,
    batch_update_hypercube_avrt,
    batch_intersect_point,
)


batch_batch_update_hypercube_avrt = torch.vmap(
    batch_update_hypercube_avrt, in_dims=(None, 0, None, 0, None, None, None)
)
batch_update_hypercubes_avrt = torch.vmap(
    batch_update_hypercube_avrt,
    in_dims=(None, 0, None, None, None, None, None),
    out_dims=(0, 0),
)
batch_batch_intersect_points = torch.vmap(batch_intersect_point)


class AVRTValidity(BaseValidity):
    def __init__(
        self,
        input_dim,
        output_dim,
        delta_frac=0.2,
        acoef=2,
        dcoef=1 / 3,
        device="cpu",
        **kwargs
    ):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.delta_frac = delta_frac
        self.acoef = acoef
        self.dcoef = dcoef

        self.previous_fb: torch.Tensor = torch.empty(
            0, 1, dtype=torch.float, device=device
        )  # (n_agents, 1) Tensor of previous feedbacks
        self.deltas: torch.Tensor = torch.empty(
            0, input_dim, 2, dtype=torch.float, device=device
        )  # (n_agents, input_dim, 2) Tensor of deltas used for avrt update
        self.orthotopes: torch.Tensor = torch.empty(
            0, input_dim, 2, device=device
        )  # (n_agents, input_dim, 2) Tensor of orthotopes

    def create(self, X, side_lengths):
        n_to_create = X.size(0)

        if n_to_create > 0:
            super().create(X, side_lengths)
            previous_fb = torch.zeros(
                (n_to_create, 1), dtype=torch.float, device=self.device
            )
            delta = batch_init_deltas_avrt(
                self.orthotopes[-n_to_create:], self.delta_frac
            )
            self.previous_fb = torch.vstack([self.previous_fb, previous_fb])
            self.deltas = torch.vstack([self.deltas, delta])

    def update(self, X, agents_mask, good, bad, no_activated=False):
        batch_size = X.size(0)
        fb_t = torch.zeros((batch_size, self.n_agents), device=X.device)
        fb_t = torch.where(
            agents_mask & no_activated.view(batch_size, 1) & good, +1, fb_t
        )
        fb_t = torch.where(
            agents_mask & ~no_activated.view(batch_size, 1) & bad, -1, fb_t
        )

        updated_hypercubes, updated_deltas = batch_batch_update_hypercube_avrt(
            self.orthotopes,  # (n_agents, n_dim, 2)
            X,  # (batch_size, n_dim)
            self.deltas,  # (n_agents, n_dim, 2)
            fb_t,  # (batch_size, n_agents)
            self.previous_fb,  # (n_agents,)
            self.acoef,  # (1,)
            self.dcoef,  # (1,)
        )  # (batch_size, n_agents, in_dim, 2),

        deltas_deltas = updated_deltas - self.deltas
        deltas_deltas = deltas_deltas.mean(dim=0)
        self.deltas += deltas_deltas
        hypercube_deltas = updated_hypercubes - self.orthotopes
        hypercube_deltas = hypercube_deltas.mean(dim=0)
        self.orthotopes += hypercube_deltas

    def immediate_expandable(self, X):
        expanded_neighbors, _ = batch_update_hypercubes_avrt(
            self.orthotopes,
            X,
            self.deltas,
            torch.ones((self.n_agents, 1)),
            self.previous_fb,
            self.acoef,
            self.dcoef,
        )  # (batch_size, n_agents, in_dim, 2) expanded shape of agents
        expanded_mask = batch_batch_intersect_points(
            expanded_neighbors,  # (batch_size*n_agents, in_dim, 2)
            X,  # (batch_size, in_dim)
        )  # (batch_size, n_agents)
        return expanded_mask
