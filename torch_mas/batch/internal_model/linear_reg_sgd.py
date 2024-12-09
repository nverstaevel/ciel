import torch
import copy
from .model_interface import InternalModelInterface
from ...common.models.linear_models import (
    batch_fit_linear_regression,
    batch_predict_linear_regression,
)


class LinearSGD(InternalModelInterface):
    def __init__(self, input_dim, output_dim, lr, device="cpu", **kwargs):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        self._sizes: torch.Tensor = torch.empty(
            0, 1, dtype=torch.long, device=device
        )  # (n_agents, 1) Tensor of fill  levels
        self.models: torch.Tensor = torch.empty(
            0,
            input_dim + 1,
            output_dim,
            dtype=torch.float,
            requires_grad=False,
            device=device,
        )  # (n_agents, input_dim+1, output_dim) Tensor of linear models

    @property
    def n_agents(self):
        return self.models.size(0)

    def create(self, X):
        batch_size = X.size(0)
        models = torch.zeros(
            (batch_size, self.input_dim + 1, self.output_dim), device=self.device
        )
        _size = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)

        self.models = torch.vstack([self.models, models])
        self._sizes = torch.vstack([self._sizes, _size])

    def maturity(self, agents_mask):
        return self._sizes[agents_mask] > (self.input_dim + 1)

    def __call__(self, X, agents_mask=None):
        return batch_predict_linear_regression(X, self.models[agents_mask])

    def clone(self):
        cloned_self = copy.copy(self)  # shallow copy
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(cloned_self, attr_name, attr_value.clone())
        return cloned_self

    def update(self, X, y, agent_mask):
        batch_size = X.size(0)
        nb_to_add_per_agent = agent_mask.sum(-1)
        agents_to_update = nb_to_add_per_agent > 0
        n_agents_to_update = agents_to_update.sum()
        self._sizes += nb_to_add_per_agent.view(self._sizes.size())

        weights = (
            agent_mask[agents_to_update].float().view(n_agents_to_update, batch_size, 1)
        )

        parameters = self.models[agents_to_update].detach()
        parameters.requires_grad_(True)

        # gradient descent step
        y_pred = batch_predict_linear_regression(
            X, parameters
        )  # (n_agents, batch_size, output_dim)
        loss = ((y_pred - y) ** 2 * weights).mean(dim=-1)  # (n_agents, batch_size)
        loss.sum().backward()

        parameters.data -= self.lr * parameters.grad.data
        parameters.grad.data.zero_()

        parameters.requires_grad_(False)
        self.models[agents_to_update] = parameters
