import torch

from abc import ABC, abstractmethod
from .activation_function import ValidityInterface
from .internal_model import InternalModelInterface


class LearningRule(ABC):
    @abstractmethod
    def __call__(
        self,
        X: torch.Tensor,
        validity: ValidityInterface,
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
        n_agents = validity.n_agents
        # solve incompetence 1
        mask_inc1 = (n_activated == 0) & (n_neighbors == 0)  # (batch_size,)
        agents_to_create = mask_inc1  # which points to use to create new agents
        activation_to_update = torch.zeros((n_agents, batch_size), dtype=torch.bool)
        models_to_update = torch.zeros((n_agents, batch_size), dtype=torch.bool)
        return (
            agents_to_create,
            activation_to_update,
            models_to_update,
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
            return (
                agents_to_create,
                activation_to_update,
                models_to_update,
            )
        return (
            torch.zeros(batch_size, dtype=torch.bool),
            torch.zeros((n_agents, batch_size), dtype=torch.bool),
            torch.zeros((n_agents, batch_size), dtype=torch.bool),
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
        n_agents = validity.n_agents
        # solve inaccuracy
        mask_inac = n_activated > 0
        if validity.n_agents > 0:
            activation_to_update = mask_inac & activated.T
            models_to_update = activated.T & (mask_inac & (~bad & ~good) | (~maturity))
            agents_to_create = torch.zeros(batch_size, dtype=torch.bool)

            return (
                agents_to_create,
                activation_to_update,
                models_to_update,
            )
        return (
            torch.zeros(batch_size, dtype=torch.bool),
            torch.zeros((n_agents, batch_size), dtype=torch.bool),
            torch.zeros((n_agents, batch_size), dtype=torch.bool),
        )
