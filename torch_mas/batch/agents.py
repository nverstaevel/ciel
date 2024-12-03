import torch

from typing import Callable
from .activation_function import ValidityInterface
from .internal_model import InternalModelInterface
from ..common.orthotopes.base import batch_intersect_points


def mse_loss(y_pred: torch.FloatTensor, y: torch.FloatTensor):
    """Calculate the mean squared error

    Args:
        y_pred (FloatTensor): (n_predictions, output_dim)
        y (FloatTensor): (output_dim,)

    Returns:
        Tensor: (n_predictions, 1)
    """
    return ((y_pred - y) ** 2).mean(dim=-1)


class AgentsTrainer:
    def __init__(
        self,
        validity: ValidityInterface,
        internal_model: InternalModelInterface,
        R: list | float,
        imprecise_th: float,
        bad_th: float,
        criterion: Callable = mse_loss,
        n_epochs: int = 10,
        batch_size: int = 64,
        device="cpu",
    ):
        self.validity = validity
        self.internal_model = internal_model
        self.criterion = criterion

        if isinstance(R, float):
            R = [R]
        self.R = torch.as_tensor(R, device=device)
        self.neighborhood_sides = torch.as_tensor(self.R, device=device)
        self.imprecise_th = imprecise_th
        self.bad_th = bad_th
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

    @property
    def n_agents(self):
        return self.validity.n_agents

    def create_agents(self, X, agents_to_create, side_lengths):
        """Create agents

        Args:
            X (Tensor): (batch_size, n_dim)
            agents_to_create (BoolTensor): (batch_size,)
            side_lengths (Tensor): (batch_size, n_dim)

        Returns:
            BoolTensor: (n_created, batch_size,)
        """
        batch_size = X.size(0)
        lows = X - side_lengths / 2
        highs = X + side_lengths / 2
        hypercubes = torch.stack([lows, highs], dim=-1)  # (batch_size,)
        agents_mask = (
            batch_intersect_points(hypercubes, X) & agents_to_create
        )  # (batch_size, n_hypercubes)

        # reduce number of created if possible
        covered_mask = torch.where(agents_to_create, False, True)  # (batch_size,)
        selected_mask = torch.zeros(batch_size, dtype=torch.bool)
        while not covered_mask.all():
            new_covered_mask = agents_mask | covered_mask  # (batch_size, n_hypercubes)
            n_newly_covered = torch.sum(
                new_covered_mask ^ covered_mask, dim=-1
            )  # (batch_size,)
            selected_id = torch.argmax(n_newly_covered, dim=-1)
            selected_mask[selected_id] = True
            covered_mask = new_covered_mask[selected_id]

        agents_to_create = selected_mask
        n_created = agents_to_create.sum()

        self.validity.create(X[agents_to_create], side_lengths[agents_to_create])
        self.internal_model.create(X[agents_to_create])

        models_to_init = torch.zeros(
            (n_created, batch_size), dtype=torch.bool
        )  # (n_created, batch_size)
        models_to_init = agents_mask[agents_to_create]
        return models_to_init

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor):
        batch_size = X.size(0)

        neighbors = self.validity.neighbors(
            X, self.neighborhood_sides
        )  # (batch_size, n_agents)
        n_neighbors = torch.count_nonzero(neighbors, dim=-1)  # (batch_size,)
        activated = self.validity.activated(X)  # (batch_size, n_agents)
        n_activated = torch.count_nonzero(activated, dim=-1)  # (batch_size,)
        maturity = self.internal_model.maturity(
            torch.ones(self.n_agents, dtype=torch.bool)
        )  # (n_agents, 1)

        agents_to_predict = neighbors.T.sum(-1) > 0
        predictions = self.internal_model(
            X, agents_to_predict
        )  # (n_agents_to_predict, batch_size, out_dim)
        propositions = torch.zeros(
            (self.n_agents, batch_size, predictions.size(-1)),
            device=self.device,
        )  # (n_agents, batch_size, out_dim)
        propositions[agents_to_predict] = predictions
        scores = self.criterion(propositions, y)  # (n_agents, batch_size)

        good = scores <= self.imprecise_th  # (n_agents, batch_size)
        bad = scores > self.bad_th  # (n_agents, batch_size)

        models_to_update = torch.zeros(
            (self.n_agents, batch_size),
            dtype=torch.bool,
            device=self.device,
        )  # (n_agents, batch_size) batch points to use to update each agent
        hypercubes_to_update = torch.zeros(
            (self.n_agents, batch_size),
            dtype=torch.bool,
            device=self.device,
        )  # (n_agents, batch_size)
        agents_to_create = torch.zeros(
            (batch_size,), dtype=torch.bool, device=self.device
        )  # (batch_size,)

        # solve incompetence 1
        mask_inc1 = (n_activated == 0) & (n_neighbors == 0)  # (batch_size,)
        agents_to_create |= mask_inc1  # which points to use to create new agents

        if self.n_agents > 0:
            # solve incompetence 2
            mask_inc2 = (n_activated == 0) & (n_neighbors > 0)  # (batch_size,)
            immediate_expandables = self.validity.immediate_expandable(
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
            self.validity.update(X, hypercubes_to_update.T, good.T, bad.T, no_activated)

        # create new agents
        # TODO: set initial agent size to be the mean of neighbors
        models_to_init = self.create_agents(
            X, agents_to_create, self.R.repeat(X.size(0), 1)
        )
        # add models to init to update
        models_to_update = torch.vstack([models_to_update, models_to_init])
        # update models
        self.internal_model.update(X, y, models_to_update)

    def fit(self, dataset):
        n_samples = len(dataset)
        for _ in range(self.n_epochs):
            indices = torch.arange(n_samples)
            shuffled_indices = indices[torch.randperm(indices.size(0))]
            batches = shuffled_indices.split(self.batch_size)
            for batch in batches:
                X, y = dataset[batch]
                self.partial_fit(X, y)

    def predict(self, X: torch.Tensor):
        """Make a prediction

        Args:
            X (Tensor): (batch_size, input_dim)

        Returns:
            Tensor: (batch_size, output_dim)
        """
        batch_size = X.size(0)
        agents_mask = torch.ones(self.n_agents, dtype=torch.bool, device=self.device)
        neighbor_mask = self.validity.neighbors(X, self.neighborhood_sides)
        maturity_mask = self.internal_model.maturity(agents_mask)
        distances = self.validity.dist_to_border(X, agents_mask)
        closest_mask = (
            torch.zeros_like(distances, dtype=torch.bool)
            .scatter(1, distances.argsort()[:, :3], True)
            .unsqueeze(-1)
        )
        mask = (neighbor_mask) & maturity_mask.T
        y_hat = self.internal_model(X, agents_mask).transpose(0, 1)

        W = mask.float().unsqueeze(-1)
        nan_mask = ~(mask.any(dim=-1))  # check if no agents are selected
        W[nan_mask] = closest_mask[nan_mask].float()

        return (y_hat * W).sum(1) / W.sum(1)
