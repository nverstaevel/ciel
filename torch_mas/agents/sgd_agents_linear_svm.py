import torch
from torch_mas.sgd_linear_svm_models import (
    batch_fit_linear_svm,
    batch_predict_linear_svm,
)
from torch_mas.hypercubes import *

from . import Agents


class AgentsLSVM(Agents):
    def __init__(self, input_dim, output_dim, memory_length, alpha, l1=0.1) -> None:
        super().__init__(input_dim, output_dim, memory_length, alpha, l1)

        self.models: torch.Tensor = torch.empty(
            0, input_dim + 1, output_dim, dtype=torch.float, requires_grad=False
        )  # (n_agents, input_dim+1, output_dim) Tensor of linear models

        self.base_prediction: torch.Tensor = torch.zeros(
            0, dtype=torch.int, requires_grad=False
        )  # (n_agents) Tensor of base prediction 

    def create_agents(self, X, side_lengths):
        """Create agents

        Args:
            X_idxs (Tensor): (batch_size, n_dim)
            side_lengths (Tensor): (batch_size, n_dim)

        Returns:
            LongTensor: (batch_size,)
        """
        batch_size = X.size(0)
        lows = X - side_lengths / 2
        highs = X + side_lengths / 2

        hypercubes = torch.stack([lows, highs], dim=-1)
        models = torch.zeros((batch_size, self.input_dim + 1, self.output_dim))
        base_prediction = torch.zeros((1))
        feature_memories = torch.zeros(
            (batch_size, self.memory_length, self.input_dim), dtype=torch.float
        )
        target_memories = torch.zeros(
            (batch_size, self.memory_length, self.output_dim), dtype=torch.float
        )
        memory_size = torch.zeros((batch_size, 1), dtype=torch.long)
        memory_ptr = torch.zeros((batch_size, 1), dtype=torch.long)

        created_idxs = torch.arange(0, batch_size) + self.hypercubes.size(0)

        self.hypercubes = torch.vstack([self.hypercubes, hypercubes])
        self.models = torch.vstack([self.models, models])
        self.base_prediction = torch.cat([self.base_prediction, base_prediction])
        self.feature_memories = torch.vstack([self.feature_memories, feature_memories])
        self.target_memories = torch.vstack([self.target_memories, target_memories])
        self.memory_sizes = torch.vstack([self.memory_sizes, memory_size])
        self.memory_ptr = torch.vstack([self.memory_ptr, memory_ptr])
        return created_idxs.long()

    def destroy_agents(self, idxs):
        """Destroy agents

        Args:
            idxs (LongTensor | BoolTensor): (batch_size,)
        """
        mask = torch.ones(self.hypercubes.shape[0], dtype=torch.bool)
        mask[idxs] = 0
        self.hypercubes = self.hypercubes[mask]
        self.feature_memories = self.feature_memories[mask]
        self.target_memories = self.target_memories[mask]
        self.models = self.models[mask]
        self.memory_sizes = self.memory_sizes[mask]
        self.memory_sizes = self.memory_ptr[mask]
        self.base_prediction = self.base_prediction[mask]

    def update_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        agents_idxs: torch.LongTensor | torch.BoolTensor,
    ):
        """Update the local model of specified agents.

        Args:
            X (Tensor): (1, input_dim)
            y (Tensor): (1, output_dim)
            agents_idxs (LongTensor | BoolTensor): (n_agents,)
        """
        # add X_idx indexes to memories
        idxs_to_add = (self.memory_ptr[agents_idxs] % self.memory_length).squeeze(-1)
        self.feature_memories[agents_idxs, idxs_to_add] = X
        self.target_memories[agents_idxs, idxs_to_add] = y
        self.memory_sizes[agents_idxs] += torch.where(
            self.memory_sizes[agents_idxs] < self.memory_length, 1, 0
        )
        self.memory_ptr[agents_idxs] += 1

        # extract memories
        X = self.feature_memories[agents_idxs]
        y = self.target_memories[agents_idxs]

        base_prediction = torch.zeros((self.base_prediction[agents_idxs].size(0)))
        models = torch.zeros((self.models[agents_idxs].size(0),X.size(2) + 1,1))
        
        has_different_classes = ((y == -1).any(dim=1) & (y == 1).any(dim=1)).reshape(X.size(0))

        # when agents have only 1 class in memory
        base_prediction[~has_different_classes] = y[~has_different_classes][:,0].squeeze()

        # when agents have all classes in memory
        if has_different_classes.sum() > 0:
            X_train = X[has_different_classes]
            y_train = y[has_different_classes]

            # update agents
            updated_models = batch_fit_linear_svm(
                X_train, y_train,
            )

            models[has_different_classes] = updated_models.unsqueeze(-1)

        self.base_prediction[agents_idxs] = base_prediction
        self.models[agents_idxs] = models

    def update_hypercube(
        self,
        X: torch.Tensor,
        agents_idxs: torch.LongTensor | torch.BoolTensor,
        good: torch.BoolTensor,
        bad: torch.BoolTensor,
        no_activated: bool = False,
    ):
        """Update hypercube of sepcified agents.

        Args:
            X (Tensor): (1, input_dim,)
            agents_idxs (LongTensor | BoolTensor): (n_agents_to_update,) | (n_agents,)
            good (BoolTensor): (n_agents_to_update,) | (n_agents,)
            bad (BoolTensor): (n_agents_to_update,) | (n_agents,)
            no_activated (bool): True if at least 1 agent activated by X
        """
        n_agents = (
            agents_idxs.size(0)
            if isinstance(agents_idxs, torch.LongTensor)
            else agents_idxs.count_nonzero()
        )
        alphas = torch.zeros((n_agents, 1))
        if no_activated:
            alphas[~bad] = self.alpha
        else:
            alphas[bad] = -self.alpha

        updated_hypercubes = batch_update_hypercube(
            self.hypercubes[agents_idxs], X.squeeze(0), alphas
        )
        self.hypercubes[agents_idxs] = updated_hypercubes

    def predict(
        self, X: torch.FloatTensor, agents_idxs: torch.LongTensor | torch.BoolTensor
    ):
        """Retrieve predictions from agents

        Args:
            X (FloatTensor): (batch_size, input_dim)
            agents_idxs (LongTensor | BoolTensor): (n_agents,)

        Returns:
            Tensor: (n_agents, batch_size, output_dim)
        """

        y_pred = torch.empty((self.models[agents_idxs].size(0),X.size(0),1))
        mask = self.base_prediction[agents_idxs] == 0

        # when agents have only 1 class in memory
        base_prediction_agents = self.base_prediction[agents_idxs]
        y_pred[~mask] = base_prediction_agents[~mask].unsqueeze(1).unsqueeze(2).repeat(1, X.size(0), 1)

        # when agents have only all classes in memory
        models_agents = self.models[agents_idxs]   
        if models_agents[mask].size(0) > 0:
            y_pred[mask] = batch_predict_linear_svm(X.unsqueeze(0), models_agents[mask])

        return y_pred

    def __call__(
        self, X: torch.FloatTensor, neighborhood_sides: torch.FloatTensor
    ) -> torch.Any:
        batch_size = X.size(0)
        agents_mask = torch.ones(self.n_agents, dtype=torch.bool)
        neighborhoods = batch_create_hypercube(
            X,
            neighborhood_sides.expand((batch_size,) + neighborhood_sides.size()),
        )
        neighborhood_mask = batch_intersect_hypercubes(neighborhoods, self.hypercubes)
        maturity_mask = self.maturity(agents_mask)
        activated_mask = batch_intersect_points(self.hypercubes, X)
        distances = batch_dist_points_to_border(self.hypercubes, X)
        closest_mask = (
            torch.zeros_like(distances, dtype=torch.bool)
            .scatter(1, distances.argsort()[:, :3], True)
            .unsqueeze(-1)
        )
        mask = (neighborhood_mask) & maturity_mask.T

        y_hat = self.predict(X, agents_mask).transpose(0, 1).float()

        W = mask.float().unsqueeze(-1)
        nan_mask = ~(mask.any(dim=-1))  # check if no agents are selected
        W[nan_mask] = closest_mask[nan_mask].float()

        W = W.squeeze()

        y_hat[W == 0] = float('nan')
        median_values,_ = y_hat.nanmedian(dim=1)  

        return median_values
