import torch

from typing import Callable
from .activation_function import ActivationInterface
from .internal_model import InternalModelInterface
from ..common.orthotopes.base import batch_intersect_points
from .learning_rules import (
    LearningRule,
    IfActivated,
    IfNoActivated,
    IfNoActivatedAndNoNeighbors,
)
from .agents import AgentsTrainer


def mse_loss(y_pred: torch.FloatTensor, y: torch.FloatTensor):
    """Calculate the mean squared error

    Args:
        y_pred (FloatTensor): (n_predictions, output_dim)
        y (FloatTensor): (output_dim,)

    Returns:
        Tensor: (n_predictions, 1)
    """
    return ((y_pred - y) ** 2).mean(dim=-1)


def weighted_mean(x, weights, dim):
    """
    Compute the weighted mean.
    Args:
        x (torch.Tensor): Tensor of values.
        weights (torch.Tensor): Tensor of weights.
    Returns:
        torch.Tensor: Weighted mean.
    """
    return (x * weights).sum(dim=dim) / weights.sum(dim=dim)


def weighted_std(x, weights, dim):
    """
    Compute the weighted standard deviation.
    Args:
        x (torch.Tensor): Tensor of values.
        weights (torch.Tensor): Tensor of weights.
    Returns:
        torch.Tensor: Weighted standard deviation.
    """
    mean = weighted_mean(x, weights, dim=dim).unsqueeze(-1)
    variance = ((x - mean) ** 2 * weights).sum(dim=dim) / weights.sum(dim=dim)
    return torch.sqrt(variance)


class ConsensusAgentsTrainer(AgentsTrainer):
    def __init__(
        self,
        validity: ActivationInterface,
        internal_model: InternalModelInterface,
        R: list | float,
        outlier_factor: float,
        learning_rules: list[LearningRule] = [
            IfNoActivatedAndNoNeighbors(),
            IfNoActivated(),
            IfActivated(),
        ],
        criterion: Callable = mse_loss,
        n_epochs: int = 10,
        batch_size: int = 64,
        device="cpu",
    ):
        self.validity = validity
        self.internal_model = internal_model
        self.learning_rules = learning_rules
        self.criterion = criterion

        if isinstance(R, float):
            R = [R]
        self.R = torch.as_tensor(R, device=device)
        self.outlier_factor = outlier_factor
        self.neighborhood_sides = torch.as_tensor(self.R, device=device)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

    # def feedbacks(self, propositions, scores, neighbors, n_neighbors):
    #     good = torch.zeros_like(scores, dtype=torch.bool)
    #     bad = torch.zeros_like(scores, dtype=torch.bool)
    #     if self.n_agents > 0:
    #         scores[~neighbors] = torch.nan
    #         scores_argmin = scores.argmin(dim=-1)
    #         scores_argmax = scores.argmax(dim=-1)
    #         good[torch.arange(self.validity.n_agents), scores_argmin] = (
    #             1.0 * ~scores.isnan()
    #         )[
    #             torch.arange(self.validity.n_agents), scores_argmin
    #         ].bool()  # good for best performer
    #         bad[torch.arange(self.validity.n_agents), scores_argmax] = (
    #             1.0 * ~scores.isnan() * (n_neighbors > 1)
    #         )[
    #             torch.arange(self.validity.n_agents), scores_argmax
    #         ].bool()  # bad for worst performer
    #     return good, bad

    def feedbacks(self, propositions, scores, neighbors, n_neighbors):
        good = torch.zeros_like(scores, dtype=torch.bool)
        bad = torch.zeros_like(scores, dtype=torch.bool)
        if self.n_agents > 0:
            weights = neighbors.float()
            mean = weighted_mean(scores.T, weights.T, dim=-1)  # (batch_size,)
            std = weighted_std(scores.T, weights.T, dim=-1)  # (batch_size,)
            good = (scores < mean - self.outlier_factor * std).bool() & neighbors
            bad = (scores > mean + self.outlier_factor * std).bool() & neighbors
        return good, bad
