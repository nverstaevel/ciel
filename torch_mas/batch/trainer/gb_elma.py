import torch
import torch.nn.functional as F

from typing import Callable
from ..activation_function import SmoothActivation
from ..internal_model import InternalModelInterface
from ...common.orthotopes.base import batch_intersect_points
from .base_trainer import BaseTrainer


class GBELMATrainer:
    def __init__(
        self,
        activation: SmoothActivation,
        internal_model: InternalModelInterface,
        n_agents: int,
        R: list | float,
        lr=1e-3,
        criterion: Callable = F.mse_loss,
        n_epochs: int = 10,
        batch_size: int = 64,
        device="cpu",
    ):
        self.activation = activation
        self.internal_model = internal_model
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.n_agents = n_agents
        if isinstance(R, float):
            R = [R]
        self.side_lengths = torch.as_tensor(R, device=device)
        self.lr = lr

    @property
    def n_agents(self):
        return self.activation.n_agents

    def destroy_agents(self, agents_to_destroy):
        """Destroy Agents

        Args:
            agents_to_destroy (torch.BoolTensor): (n_agents,)
        """
        self.activation.destroy(agents_to_destroy)
        self.internal_model.destroy(agents_to_destroy)

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

        n_created = agents_to_create.sum()

        self.activation.create(X[agents_to_create], side_lengths[agents_to_create])
        self.internal_model.create(X[agents_to_create])

        models_to_init = torch.zeros(
            (n_created, batch_size), dtype=torch.bool
        )  # (n_created, batch_size)
        models_to_init = agents_mask[agents_to_create]
        return models_to_init

    def fit(self, dataset):
        n_samples = len(dataset)

        for _ in range(self.n_epochs):
            indices = torch.arange(n_samples)
            shuffled_indices = indices[torch.randperm(indices.size(0))]
            batches = shuffled_indices.split(self.batch_size)
            for batch in batches:
                b_X, b_y = dataset[batch]
                activations = self.activation.activated(b_X)
                propositions = self.internal_model(b_X)
                predictions = torch.mean(predictions * activations, dim=0)

                loss = self.criterion(predictions, b_y)

            # TODO create optimizers and update
