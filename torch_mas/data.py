import torch
import numpy as np
from torch.utils.data import Dataset


class DataBuffer(Dataset):
    def __init__(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> None:
        self.input_dim = X.shape[-1]
        self.output_dim = y.shape[-1]
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)
        assert self.X.size(0) == self.y.size(0)

    def add(self, X, y):
        self.X = torch.vstack([self.X, X])
        self.y = torch.vstack([self.y, y])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
