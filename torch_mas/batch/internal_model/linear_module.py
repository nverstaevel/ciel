import torch
import copy

from torch_mas.common.models.linear_models import batch_predict_linear_regression
from .model_interface import InternalModelInterface


class LinearModule(InternalModelInterface):
    def __init__(self, input_dim, output_dim, device="cpu", **kwargs):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.models: torch.Tensor = torch.empty(
            0,
            input_dim + 1,
            output_dim,
            dtype=torch.float,
            requires_grad=True,
            device=device,
        )  # (n_agents, input_dim+1, output_dim) Tensor of linear models

    @property
    def n_agents(self):
        return self.models.size(0)

    def destroy(self, agents_mask):
        self.models = self.models[~agents_mask]

    def create(self, X):
        batch_size = X.size(0)
        models = torch.zeros(
            (batch_size, self.input_dim + 1, self.output_dim), device=self.device
        )
        self.models = torch.vstack([self.models, models])

    def __call__(self, X, agents_mask=None):
        return batch_predict_linear_regression(X, self.models[agents_mask])

    def clone(self):
        cloned_self = copy.copy(self)  # shallow copy
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(cloned_self, attr_name, attr_value.clone())
        return cloned_self

    def parameters(self):
        return [self.models]
