import torch

from typing import Callable
from .validity import ValidityInterface
from .internal_model import InternalModelInterface
from ..common.orthotopes.base import batch_sides


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
        self.device = device

    @property
    def n_agents(self):
        return self.validity.n_agents

    def create_agents(self, X, side_lengths):
        """Create agents

        Args:
            X (Tensor): (batch_size, n_dim)
            side_lengths (Tensor): (batch_size, n_dim)

        Returns:
            BoolTensor: (n_created, batch_size,)
        """
        created_idxs = torch.arange(0, X.size(0), dtype=torch.long) + self.n_agents
        self.validity.create(X, side_lengths)
        self.internal_model.create(X)

        return created_idxs

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor):
        neighborhood_agents = self.validity.neighbors(X, self.neighborhood_sides)
        n_neighbors = torch.count_nonzero(neighborhood_agents)
        activated_agents = self.validity.activated(X.squeeze(0))
        n_activated = torch.count_nonzero(activated_agents)
        agents_to_update = torch.empty(0, device=self.device)
        if n_activated == 0 and n_neighbors == 0:
            created_idxs = self.create_agents(X, self.R)
            agents_to_update = torch.concat([agents_to_update, created_idxs])

        if n_activated == 0 and n_neighbors > 0:
            expanded_mask = self.validity.immediate_expandable(X, neighborhood_agents)
            expanded_idxs = torch.arange(self.n_agents, device=self.device)[
                neighborhood_agents
            ][expanded_mask]
            activated_maturity = self.internal_model.maturity(expanded_idxs).squeeze(-1)
            expanded_idxs = expanded_idxs[activated_maturity]
            n_expand_candidates = len(expanded_idxs)
            if n_expand_candidates > 0:
                predictions = self.internal_model(X, expanded_idxs)
                score = self.criterion(predictions, y).squeeze(-1)  # (n_predictions,)
                good = score <= self.imprecise_th
                bad = score > self.bad_th

                self.validity.update(X, expanded_idxs, good, bad, no_activated=True)

                agents_to_update = torch.arange(self.n_agents, device=self.device)[
                    expanded_idxs
                ][~bad & ~good]
                if bad.all():
                    created_idxs = self.create_agents(X, self.R)
                    agents_to_update = torch.concat([agents_to_update, created_idxs])
            else:
                radius = self.R
                if n_neighbors > 1:
                    radius = batch_sides(
                        self.validity.orthotopes[neighborhood_agents]
                    ).mean(0)
                created_idxs = self.create_agents(X, radius)
                agents_to_update = torch.concat([agents_to_update, created_idxs])
        if n_activated > 0:
            agents_mask = activated_agents
            predictions = self.internal_model(X, agents_mask)
            score = self.criterion(predictions, y).squeeze(-1)  # (n_predictions,)
            activated_maturity = self.internal_model.maturity(agents_mask).squeeze(-1)

            good = score <= self.imprecise_th
            bad = score > self.bad_th

            self.validity.update(X, agents_mask, good, bad, no_activated=False)

            agents_to_update = torch.arange(self.n_agents, device=self.device)[
                agents_mask
            ][~bad & ~good | ~activated_maturity]
        if agents_to_update.size(0) > 0:
            self.internal_model.update(X, y, agents_to_update.long())

    def fit(self, dataset):
        n_samples = len(dataset)
        for _ in range(self.n_epochs):
            indices = torch.arange(n_samples)
            shuffled_indices = indices[torch.randperm(indices.size(0))]
            batches = shuffled_indices.split(1)
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
