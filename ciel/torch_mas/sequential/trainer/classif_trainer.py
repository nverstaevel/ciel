import torch

from . import BaseTrainer
from ..activation_function import ActivationInterface


class ClassifTrainer(BaseTrainer):

    def predict(self, X: torch.Tensor):
        batch_size = X.size(0)
        agents_mask = torch.ones(self.n_agents, dtype=torch.bool, device=self.device)
        neighbor_mask = self.activation.neighbors(X, self.neighborhood_sides)
        maturity_mask = self.internal_model.maturity(agents_mask)
        distances = self.activation.dist_to_border(X, agents_mask)
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

        W = W.squeeze()

        y_hat = y_hat.float()
        y_hat[W == 0] = torch.nan

        median_values,_ = y_hat.nanmedian(dim=1)  

        return median_values

