import torch
import copy
from .model_interface import InternalModelInterface
from ...common.models.svm_sgd import (
    batch_fit_linear_svm,
    batch_predict_linear_svm,
)


class SVM(InternalModelInterface):
    def __init__(
        self, input_dim, output_dim, memory_length, device="cpu", **kwargs
    ):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memory_length = memory_length

        self.feature_memories: torch.Tensor = torch.empty(
            0, memory_length, input_dim, dtype=torch.float, device=device
        )  # (n_agents, memory_length,) Tensor of features
        self.target_memories: torch.Tensor = torch.empty(
            0, memory_length, output_dim, dtype=torch.float, device=device
        )  # (n_agents, memory_length,) Tensor of targets
        self.memory_sizes: torch.Tensor = torch.empty(
            0, 1, dtype=torch.long, device=device
        )  # (n_agents, 1) Tensor of fill memory levels
        self.memory_ptr: torch.Tensor = torch.empty(
            0, 1, dtype=torch.long, device=device
        )  # (n_agents, 1) Tensor of fill memory levels
        
        self.models: torch.Tensor = torch.empty(
            0, input_dim + 1, output_dim, dtype=torch.float, requires_grad=False, device=device
        )  # (n_agents, input_dim+1, output_dim) Tensor of linear models

        self.base_prediction: torch.Tensor = torch.zeros(
            0, dtype=torch.float, requires_grad=False, device=device
        )  # (n_agents) Tensor of base prediction 

    @property
    def n_agents(self):
        return self.models.size(0)

    def create(self, X):
        batch_size = X.size(0)
        models = torch.zeros(
            (batch_size, self.input_dim + 1, self.output_dim), device=self.device
        )
        base_prediction = torch.zeros((batch_size,), dtype=torch.float, device=self.device)
        feature_memories = torch.zeros(
            (batch_size, self.memory_length, self.input_dim),
            dtype=torch.float,
            device=self.device,
        )
        target_memories = torch.zeros(
            (batch_size, self.memory_length, self.output_dim),
            dtype=torch.float,
            device=self.device,
        )
        memory_size = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        memory_ptr = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)

        self.models = torch.vstack([self.models, models])
        self.base_prediction = torch.vstack([self.base_prediction.unsqueeze(-1), base_prediction.unsqueeze(-1)]).squeeze(-1)
        self.feature_memories = torch.vstack([self.feature_memories, feature_memories])
        self.target_memories = torch.vstack([self.target_memories, target_memories])
        self.memory_sizes = torch.vstack([self.memory_sizes, memory_size])
        self.memory_ptr = torch.vstack([self.memory_ptr, memory_ptr])

    def maturity(self, agents_mask):
        return self.memory_sizes[agents_mask] > (self.input_dim + 1)

    def update(self, X, y, agent_mask):
        # add X_idx indexes to memories
        idxs_to_add = (self.memory_ptr[agent_mask] % self.memory_length).squeeze(-1)
        self.feature_memories[agent_mask, idxs_to_add] = X
        self.target_memories[agent_mask, idxs_to_add] = y
        self.memory_sizes[agent_mask] += torch.where(
            self.memory_sizes[agent_mask] < self.memory_length, 1, 0
        )
        self.memory_ptr[agent_mask] += 1

        # extract memories
        X = self.feature_memories[agent_mask]
        y = self.target_memories[agent_mask]

        base_prediction = torch.zeros(self.base_prediction[agent_mask].size(0), dtype=torch.float, device=self.device)
        models = torch.zeros((self.models[agent_mask].size(0),X.size(2) + 1,1), device=self.device)
        
        has_different_classes = ((y == -1).any(dim=1) & (y == 1).any(dim=1)).reshape(X.size(0))

        # when agents have only 1 class in memory
        base_prediction[~has_different_classes] = y[~has_different_classes][:,0].squeeze()

        # when agents have all classes in memory
        if has_different_classes.sum() > 0:
            X_train = X[has_different_classes]
            y_train = y[has_different_classes]

            # update agents
            updated_models = batch_fit_linear_svm(
                X_train, y_train, device=self.device
            )

            models[has_different_classes] = updated_models.unsqueeze(-1)

        self.base_prediction[agent_mask] = base_prediction
        self.models[agent_mask] = models

    def __call__(self, X, agents_mask=None):
        y_pred = torch.empty((self.models[agents_mask].size(0),X.size(0),1),dtype=torch.float,device=self.device)
        mask = self.base_prediction[agents_mask] == 0

        # when agents have only 1 class in memory
        base_prediction_agents = self.base_prediction[agents_mask]
        y_pred[~mask] = base_prediction_agents[~mask].unsqueeze(1).unsqueeze(2).repeat(1, X.size(0), 1)

        # when agents have only all classes in memory
        models_agents = self.models[agents_mask]   
        if models_agents[mask].size(0) > 0:
            y_pred[mask] = batch_predict_linear_svm(X.unsqueeze(0), models_agents[mask])

        return y_pred


    def clone(self):
        cloned_self = copy.copy(self)  # shallow copy
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(cloned_self, attr_name, attr_value.clone())
        return cloned_self
