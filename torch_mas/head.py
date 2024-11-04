import torch
import numpy as np

from torch_mas.agents import Agents
from torch_mas.orthotopes.hypercubes import *
from torch_mas.models.linear_models import *


class Head:
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
        l1 = 0.0,
        random_state = None,
        verbose = False,
        device = "cpu"
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
            random_state (optional): seed the RNG 
            verbose (boolean, optional): verbose option
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        if isinstance(R, float):
            R = [R]
        self.R = torch.tensor(R, dtype=torch.float32, device=device)
        self.neighborhood_sides = torch.as_tensor(self.R, device=device)
        self.imprecise_th = imprecise_th
        self.bad_th = bad_th
        self.alpha = alpha
        self.memory_length = memory_length
        self.n_epochs = n_epochs
        self.l1_penalty = l1
        self.random_state = random_state
        self.verbose = verbose
        self.device = device

        self.agents = agents(
            self.input_dim,
            self.output_dim,
            self.memory_length,
            self.alpha,
            l1=self.l1_penalty,
            device=device
        )

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

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
        neighborhood_agents = self.agents.neighbors(X, self.neighborhood_sides)
        n_neighbors = torch.count_nonzero(neighborhood_agents)
        activated_agents = self.agents.activated(X.squeeze(0))
        n_activated = torch.count_nonzero(activated_agents)
        agents_to_update = torch.empty(0,device=self.device)
        if n_activated == 0 and n_neighbors == 0:
            created_idxs = self.agents.create_agents(X, self.R)
            agents_to_update = torch.concat([agents_to_update, created_idxs])

        if n_activated == 0 and n_neighbors > 0:
            expanded_mask = self.agents.immediate_expandable(X, neighborhood_agents)
            expanded_idxs = torch.arange(self.agents.n_agents, device=self.device)[neighborhood_agents][
                expanded_mask
            ]
            activated_maturity = self.agents.maturity(expanded_idxs).squeeze(-1)
            expanded_idxs = expanded_idxs[activated_maturity]
            n_expand_candidates = len(expanded_idxs)
            if n_expand_candidates > 0:
                predictions = self.agents.predict(X, expanded_idxs)
                score = self.score(predictions, y).squeeze(-1)  # (n_predictions,)
                good = score <= self.imprecise_th
                bad = score > self.bad_th

                self.agents.update_hypercube(
                    X, expanded_idxs, good, bad, no_activated=True
                )

                agents_to_update = torch.arange(self.agents.n_agents, device=self.device)[expanded_idxs][
                    ~bad & ~good
                ]
                if bad.all():
                    created_idxs = self.agents.create_agents(X, self.R)
                    agents_to_update = torch.concat([agents_to_update, created_idxs])
            else:
                radius = self.R
                if n_neighbors > 1:
                    radius = batch_sides(
                        self.agents.hypercubes[neighborhood_agents]
                    ).mean(0)
                created_idxs = self.agents.create_agents(X, radius)
                agents_to_update = torch.concat([agents_to_update, created_idxs])
        if n_activated > 0:
            agents_mask = activated_agents
            predictions = self.agents.predict(X, agents_mask)
            score = self.score(predictions, y).squeeze(-1)  # (n_predictions,)
            activated_maturity = self.agents.maturity(agents_mask).squeeze(-1)

            good = score <= self.imprecise_th
            bad = score > self.bad_th

            self.agents.update_hypercube(X, agents_mask, good, bad, no_activated=False)

            agents_to_update = torch.arange(self.agents.n_agents, device=self.device)[agents_mask][
                ~bad & ~good | ~activated_maturity
            ]
        if agents_to_update.size(0) > 0:
            self.agents.update_model(X, y, agents_to_update.long())

    def fit(self, dataset):
        n_samples = len(dataset)
        idxs = np.arange(0, n_samples)
        np.random.shuffle(idxs)

        self._step = 0
        for e in range(self.n_epochs):
            for idx in idxs:
                X, y = dataset[torch.LongTensor([idx])]
                self.partial_fit(X, y)
                self._step += 1

            if self.verbose:
                print(f'Epoch {e}: {self.agents.models.size(0)} agents')
            

    def predict(self, X):
        """Make a prediction

        Args:
            X (Tensor): (batch_size, input_dim)

        Returns:
            Tensor: (batch_size, output_dim)
        """
        return self.agents(X, self.neighborhood_sides)
