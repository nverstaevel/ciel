import torch
import numpy as np

from torch_mas.agents import Agents
from torch_mas.agents.batch_agents_linear_reg import BatchLinearAgent
from torch_mas.orthotopes.hypercubes import *
from torch_mas.models.linear_models import *


class AlternativeBatchHead:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        R: list | float,
        agents: Agents,
        agents_kwargs={},
        n_epochs: int = 10,
        batch_size=64,
        device="cpu",
    ) -> None:
        """Initialize the learning algorithm.

        Args:
            input_dim (int): size of the input vector.
            output_dim (int): size of the output vector.
            R (list | float): size of the sidelengths of a newly created agent. If R is a list then each value should correspond to a dimension of the input vector.
            imprecise_th (float): absolute threshold below which an agent's proposition is considered good.
            bad_th (float): absolute threshold above which an agent's proposition is considered bad.
            agents (Agents): type of agents. It must heritate of the Agents class.
            agents_kwargs (dict): parameters of agents.
            n_epochs (int, optional): number of times each data point is seen by the agents during learning. Defaults to 10.
            device (str, optional): chose device on which calculations are done (cpu or cuda). Default to cpu.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        if isinstance(R, float):
            R = [R]
        self.R = torch.as_tensor(R, device=device)
        self.neighborhood_sides = torch.as_tensor(self.R, device=device)
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.agents: BatchLinearAgent = agents(
            self.input_dim, self.output_dim, device=device, **agents_kwargs
        )

    def score(self, y_pred: torch.FloatTensor, y: torch.FloatTensor):
        """Calculate the mean squared error

        Args:
            y_pred (FloatTensor): (n_predictions, output_dim)
            y (FloatTensor): (output_dim,)

        Returns:
            Tensor: (n_predictions, 1)
        """
        return ((y_pred - y) ** 2).mean(dim=-1)

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor):
        batch_size = X.size(0)

        neighbors = self.agents.neighbors(
            X, self.neighborhood_sides
        )  # (batch_size, n_agents)
        n_neighbors = torch.count_nonzero(neighbors, dim=-1)  # (batch_size,)
        activated = self.agents.activated(X)  # (batch_size, n_agents)
        n_activated = torch.count_nonzero(activated, dim=-1)  # (batch_size,)
        maturity = self.agents.maturity(
            torch.ones(self.agents.n_agents, dtype=torch.bool)
        )  # (n_agents, 1)

        propositions = torch.zeros(
            (self.agents.n_agents, batch_size, self.output_dim),
            device=self.agents.device,
        )
        agents_to_predict = neighbors.T.sum(-1) > 0
        propositions[agents_to_predict] = self.agents.predict(
            X, agents_to_predict
        )  # (n_agents, batch_size, out_dim)
        scores = self.score(propositions, y)  # (n_agents, batch_size)

        models_to_update = torch.zeros(
            (self.agents.n_agents, batch_size),
            dtype=torch.bool,
            device=self.agents.device,
        )  # (n_agents, batch_size) batch points to use to update each agent
        hypercubes_to_update = torch.zeros(
            (self.agents.n_agents, batch_size),
            dtype=torch.bool,
            device=self.agents.device,
        )  # (n_agents, batch_size)
        agents_to_create = torch.zeros(
            (batch_size,), dtype=torch.bool, device=self.agents.device
        )  # (batch_size,)

        # solve incompetence 1
        mask_inc1 = (n_activated == 0) & (n_neighbors == 0)  # (batch_size,)
        agents_to_create |= mask_inc1  # which points to use to create new agents

        if self.agents.n_agents > 0:
            scores[~neighbors.T] = torch.nan
            scores_argmin = scores.argmin(dim=-1)
            scores_argmax = scores.argmax(dim=-1)
            good = torch.zeros_like(scores, dtype=torch.bool)
            bad = torch.zeros_like(scores, dtype=torch.bool)
            good[torch.arange(self.agents.n_agents), scores_argmin] = (
                1.0 * ~scores.isnan()
            )[
                torch.arange(self.agents.n_agents), scores_argmin
            ].bool()  # good for best performer
            bad[torch.arange(self.agents.n_agents), scores_argmax] = (
                1.0 * ~scores.isnan() * (n_neighbors > 1)
            )[
                torch.arange(self.agents.n_agents), scores_argmax
            ].bool()  # bad for worst performer

            # solve incompetence 2
            mask_inc2 = (n_activated == 0) & (n_neighbors > 0)  # (batch_size,)
            immediate_expandables = self.agents.immediate_expandable(
                X
            )  # (batch_size, n_agents)
            expand_candidates = (
                immediate_expandables & maturity.T
            )  # (batch_size, n_agents)
            n_expand_candidates = torch.sum(expand_candidates, dim=-1)  # (batch_size,)

            hypercubes_to_update |= mask_inc2 & expand_candidates.T
            models_to_update |= mask_inc2 & (
                (n_expand_candidates > 0) & expand_candidates.T
            )
            agents_to_create |= mask_inc2 & (n_expand_candidates == 0)

            # solve inaccuracy
            mask_inac = n_activated > 0
            hypercubes_to_update |= mask_inac & activated.T
            models_to_update |= activated.T & (mask_inac & (~bad & ~good) | (~maturity))

            # update hypercubes
            no_activated = mask_inc2 & (~mask_inac)
            self.agents.update_hypercube(
                X, hypercubes_to_update.T, good.T, bad.T, no_activated
            )

        # create new agents
        # TODO: set initial agent size to be the mean of neighbors
        models_to_init = self.agents.create_agents(
            X, agents_to_create, self.R.repeat(X.size(0), 1)
        )
        # add models to init to update
        models_to_update = torch.vstack([models_to_update, models_to_init])
        # update models
        self.agents.update_model(X, y, models_to_update)

    def fit(self, dataset):
        n_samples = len(dataset)
        self._step = 0
        self.volume_hist = []
        for _ in range(self.n_epochs):
            indices = torch.arange(n_samples)
            shuffled_indices = indices[torch.randperm(indices.size(0))]
            batches = shuffled_indices.split(self.batch_size)
            for batch in batches:
                X, y = dataset[batch]
                self.partial_fit(X, y)
                self._step += 1
                self.volume_hist += [
                    torch.sum(batch_volume(self.agents.hypercubes))
                    .cpu()
                    .detach()
                    .item()
                ]

    def predict(self, X):
        """Make a prediction

        Args:
            X (Tensor): (batch_size, input_dim)

        Returns:
            Tensor: (batch_size, output_dim)
        """
        if self.agents.n_agents < 1:  # if no agent return white noise
            return torch.randn((X.size(0), self.output_dim), device=X.device)
        return self.agents(X, self.neighborhood_sides)
