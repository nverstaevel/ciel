import torch
from sklearn.base import BaseEstimator
from torch_mas.data import DataBuffer
from torch_mas.head import Head
from torch_mas.agents import Agents


class Ciel(BaseEstimator):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        R: list | float,
        imprecise_th: float,
        bad_th: float,
        alpha: float,
        agents:Agents,
        memory_length: int = 20,
        n_epochs: int = 10,
        l1 = 0.0,
        random_state = None
    ) -> None:
        """Initialize the learning algorithm.

        Args:
            input_dim (int): size of the input vector.
            output_dim (int): size of the output vector.
            R (list | float): size of the sidelengths of a newly created agent. If R is a list then each value should correspond to a dimension of the input vector.
            imprecise_th (float): absolute threshold below which an agent's proposition is considered good.
            bad_th (float): absolute threshold above which an agent's proposition is considered bad.
            alpha (float): coefficient of expansion or retraction of agents.
            memory_length (int, optional): size of an agent's memory. Defaults to 20.
            n_epochs (int, optional): number of times each data point is seen by the agents during learning. Defaults to 10.
            l1 (float, optional): coefficient of l1 regularization. Defaults to 0.
            random_state (optional): seed the RNG 
        """
        self._estimator_type = "regressor"
        self.base_estimator = Head
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.R = R
        self.imprecise_th = imprecise_th
        self.bad_th = bad_th
        self.alpha = alpha
        self.memory_length = memory_length
        self.n_epochs = n_epochs
        self.l1 = l1
        self.agents = agents
        self.random_state = random_state

        self.estimator = Head(
            self.input_dim,
            self.output_dim,
            self.R,
            self.imprecise_th,
            self.bad_th,
            self.alpha,
            self.agents,
            self.memory_length,
            self.n_epochs,
            self.l1,
            self.random_state
        )



    def fit(self, X, y):
        return self.estimator.fit(DataBuffer(X, y))

    def predict(self, X):
        return (
            self.estimator.predict(torch.from_numpy(X).float()).detach().numpy()
        )

    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        self.estimator = Head(
            self.input_dim,
            self.output_dim,
            self.R,
            self.imprecise_th,
            self.bad_th,
            self.alpha,
            self.memory_length,
            self.n_epochs,
        )
        return self
