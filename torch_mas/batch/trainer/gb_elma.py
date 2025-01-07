import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
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
        n_agents_ini: int,
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
        self.n_agents_init = n_agents_ini
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

    def create_agents(self, X, side_lengths):
        """Create agents

        Args:
            X (Tensor): (batch_size, n_dim)
            agents_to_create (BoolTensor): (batch_size,)
            side_lengths (Tensor): (batch_size, n_dim)

        Returns:
            BoolTensor: (n_created, batch_size,)
        """
        self.activation.create(X, side_lengths)
        self.internal_model.create(X)

    def fit(self, dataset: Dataset):
        n_samples = len(dataset)
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size)

        # initialize agents
        ini_indices = torch.randperm(n_samples)[: self.n_agents_init]
        ini_X, _ = dataset[ini_indices]
        self.create_agents(ini_X, self.side_lengths)

        optimizer = torch.optim.Adam(
            self.activation.parameters() + self.internal_model.parameters(), lr=self.lr
        )

        # training loop
        for _ in range(self.n_epochs):
            for b_X, b_y in train_dataloader:
                activations = self.activation.activated(b_X).unsqueeze(-1)
                propositions = self.internal_model(b_X)
                predictions = torch.mean(propositions * activations, dim=0)

                loss = self.criterion(predictions, b_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, X):
        activations = self.activation.activated(X).unsqueeze(-1)
        propositions = self.internal_model(X)
        return torch.mean(propositions * activations, dim=0)
