import torch
import copy

from .activation_interface import ActivationInterface


def dim_activated(x: torch.Tensor, h: torch.Tensor, limit_probability: torch.Tensor):
    """Return activation for one dimension of one point for one hyperrectangle

    Args:
        x (torch.Tensor): (batch_size, in_dims)
        h (torch.Tensor): (in_dims, 2)
        limit_probability (float, optional): probability reached on limits of hyperrectangle. Defaults to 0.3.

    Returns:
        torch.Tensor: (batch_size, in_dims)
    """
    low, high = h[:, 0], h[:, 1]
    mu = (high + low) / 2
    sigma = torch.sqrt(-((high - mu) ** 2) / (2 * torch.log(limit_probability)))
    return torch.exp(-((x - mu) ** 2) / (2 * sigma**2))


batch_dim_activated = torch.vmap(
    dim_activated, in_dims=(None, 0, None)
)  # batch over hyperrectangles


def activated(x: torch.Tensor, h: torch.Tensor, limit_probability: torch.Tensor):
    """Return activation value of a batch of point given a batch of hyperrectangles

    Args:
        x (torch.Tensor): (x_batch_size, in_dims)
        h (torch.Tensor): (h_batch_size, in_dims, 2)
        limit_probability (float, optional): probability reached on limits of hyperrectangle. Defaults to 0.3.

    Returns:
        torch.Tensor: (x_batch_size)
    """
    a = batch_dim_activated(
        x, h, torch.as_tensor(limit_probability)
    )  # activation for each individual dimension
    return torch.prod(a, dim=-1)


class SmoothActivation(ActivationInterface):
    def __init__(self, input_dim, output_dim, limit_value, device="cpu"):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.limit_value = torch.as_tensor(limit_value)

        self.orthotopes: torch.Tensor = torch.empty(
            0, input_dim, 2, device=device
        )  # (n_agents, input_dim, 2) Tensor of orthotopes

    @property
    def n_agents(self):
        return self.orthotopes.size(0)

    def destroy(self, agents_mask):
        self.orthotopes = self.orthotopes[~agents_mask]

    def create(self, X, side_lengths):
        lows = X - side_lengths / 2
        highs = X + side_lengths / 2
        orthotopes = torch.stack([lows, highs], dim=-1)
        self.orthotopes = torch.vstack([self.orthotopes, orthotopes])

    def activated(self, X):
        return activated(X, self.orthotopes, self.limit_value)

    def clone(self):
        cloned_self = copy.copy(self)  # shallow copy
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(cloned_self, attr_name, attr_value.clone())
        return cloned_self

    def parameters(self):
        return [self.orthotopes]
