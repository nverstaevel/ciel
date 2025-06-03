import torch

from abc import ABC, abstractmethod
from ..activation_function import ActivationInterface
from ..internal_model import InternalModelInterface


class LearningRule(ABC):
    @abstractmethod
    def __call__(
        self,
        X: torch.Tensor,
        validity: ActivationInterface,
        internal_model: InternalModelInterface,
        good: torch.Tensor,
        bad: torch.Tensor,
        activated: torch.BoolTensor,
        neighbors: torch.BoolTensor,
        n_activated: torch.Tensor,
        n_neighbors: torch.Tensor,
        maturity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class IfNoActivatedAndNoNeighbors(LearningRule):
    def __call__(
        self,
        X,
        validity,
        internal_model,
        good,
        bad,
        activated,
        neighbors,
        n_activated,
        n_neighbors,
        maturity,
    ):
        batch_size = X.size(0)
        device = X.device

        n_agents = validity.n_agents
        # solve incompetence 1
        mask_inc1 = (n_activated == 0) & (n_neighbors == 0)  # (batch_size,)
        agents_to_create = mask_inc1  # which points to use to create new agents
        activation_to_update = torch.zeros(
            (n_agents, batch_size), dtype=torch.bool, device=device
        )
        models_to_update = torch.zeros(
            (n_agents, batch_size), dtype=torch.bool, device=device
        )
        agents_to_destroy = torch.zeros((n_agents,), dtype=torch.bool, device=device)
        return (
            agents_to_create,
            activation_to_update,
            models_to_update,
            agents_to_destroy,
        )


class IfNoActivated(LearningRule):
    def __call__(
        self,
        X,
        validity,
        internal_model,
        good,
        bad,
        activated,
        neighbors,
        n_activated,
        n_neighbors,
        maturity,
    ):
        batch_size = X.size(0)
        device = X.device

        n_agents = validity.n_agents
        # solve incompetence 2
        mask_inc2 = (n_activated == 0) & (n_neighbors > 0)  # (batch_size,)
        if validity.n_agents > 0:
            immediate_expandables = validity.immediate_expandable(
                X
            )  # (batch_size, n_agents)
            expand_candidates = (
                immediate_expandables & maturity.T
            )  # (batch_size, n_agents)
            n_expand_candidates = torch.sum(expand_candidates, dim=-1)  # (batch_size,)

            activation_to_update = mask_inc2 & expand_candidates.T
            models_to_update = mask_inc2 & (
                (n_expand_candidates > 0) & expand_candidates.T
            )
            agents_to_create = mask_inc2 & (n_expand_candidates == 0)
            agents_to_destroy = torch.zeros(n_agents, dtype=torch.bool, device=device)
            return (
                agents_to_create,
                activation_to_update,
                models_to_update,
                agents_to_destroy,
            )
        return (
            torch.zeros(batch_size, dtype=torch.bool, device=device),
            torch.zeros((n_agents, batch_size), dtype=torch.bool, device=device),
            torch.zeros((n_agents, batch_size), dtype=torch.bool, device=device),
            torch.zeros((n_agents,), dtype=torch.bool, device=device),
        )


class IfActivated(LearningRule):
    def __call__(
        self,
        X,
        validity,
        internal_model,
        good,
        bad,
        activated,
        neighbors,
        n_activated,
        n_neighbors,
        maturity,
    ):
        batch_size = X.size(0)
        device = X.device

        n_agents = validity.n_agents
        # solve inaccuracy
        mask_inac = n_activated > 0
        if validity.n_agents > 0:
            activation_to_update = mask_inac & activated.T
            models_to_update = activated.T & (mask_inac & (~bad & ~good) | (~maturity))
            agents_to_create = torch.zeros(batch_size, dtype=torch.bool, device=device)
            agents_to_destroy = torch.zeros(n_agents, dtype=torch.bool, device=device)
            return (
                agents_to_create,
                activation_to_update,
                models_to_update,
                agents_to_destroy,
            )
        return (
            torch.zeros(batch_size, dtype=torch.bool, device=device),
            torch.zeros((n_agents, batch_size), dtype=torch.bool, device=device),
            torch.zeros((n_agents, batch_size), dtype=torch.bool, device=device),
            torch.zeros((n_agents,), dtype=torch.bool, device=device),
        )


class SimpleDestroy(LearningRule):
    def __init__(self, imbalance_th=20):
        self.imbalance_th = imbalance_th

    def __call__(
        self,
        X,
        validity,
        internal_model,
        good,
        bad,
        activated,
        neighbors,
        n_activated,
        n_neighbors,
        maturity,
    ):
        batch_size = X.size(0)
        device = X.device
        n_agents = validity.n_agents
        balanced = validity.bads - validity.goods
        agents_to_destroy = (balanced > self.imbalance_th) & maturity
        return (
            torch.zeros(batch_size, dtype=torch.bool, device=device),
            torch.zeros((n_agents, batch_size), dtype=torch.bool, device=device),
            torch.zeros((n_agents, batch_size), dtype=torch.bool, device=device),
            agents_to_destroy.view(-1),
        )
