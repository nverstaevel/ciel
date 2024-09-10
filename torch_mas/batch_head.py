import torch
import numpy as np
import tqdm

from torch_mas.agents import Agents
from torch_mas.agents.batch_agents_linear_reg import BatchLinearAgent
from torch_mas.hypercubes import *
from torch_mas.linear_models import *


class BatchHead:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        R: list | float,
        imprecise_th: float,
        bad_th: float,
        alpha: float,
        agents: Agents,
        memory_length: int = 20,
        n_epochs: int = 10,
        batch_size=64,
        l1=0.0,
        device="cpu",
    ) -> None:
        """Initialize the learning algorithm.

        Args:
            input_dim (int): size of the input vector.
            output_dim (int): size of the output vector.
            R (list | float): size of the sidelengths of a newly created agent. If R is a list then each value should correspond to a dimension of the input vector.
            imprecise_th (float): absolute threshold below which an agent's proposition is considered good.
            bad_th (float): absolute threshold above which an agent's proposition is considered bad.
            alpha (float): coefficient of expansion or retraction of agents.
            agents (Agents): type of agents. It must heritate of the Agents class.
            memory_length (int, optional): size of an agent's memory. Defaults to 20.
            n_epochs (int, optional): number of times each data point is seen by the agents during learning. Defaults to 10.
            l1 (float, optional): coefficient of l1 regularization. Defaults to 0.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        if isinstance(R, float):
            R = [R]
        self.R = torch.tensor(R, device=device)
        self.neighborhood_sides = torch.tensor(self.R, device=device)
        self.imprecise_th = imprecise_th
        self.bad_th = bad_th
        self.alpha = alpha
        self.memory_length = memory_length
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l1_penalty = l1

        self.agents: BatchLinearAgent = agents(
            self.input_dim,
            self.output_dim,
            self.memory_length,
            self.alpha,
            l1=self.l1_penalty,
            device=device,
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

        good = scores <= self.imprecise_th  # (n_agents, batch_size)
        bad = scores > self.bad_th  # (n_agents, batch_size)

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
        for _ in range(self.n_epochs):
            indices = torch.arange(n_samples)
            shuffled_indices = indices[torch.randperm(indices.size(0))]
            batches = shuffled_indices.split(self.batch_size)
            for batch in batches:
                X, y = dataset[batch]
                self.partial_fit(X, y)
                self._step += 1

    def predict(self, X):
        """Make a prediction

        Args:
            X (Tensor): (batch_size, input_dim)

        Returns:
            Tensor: (batch_size, output_dim)
        """
        return self.agents(X, self.neighborhood_sides)
