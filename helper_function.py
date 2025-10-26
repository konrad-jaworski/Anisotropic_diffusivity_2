import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pyDOE2 import lhs

class DomainDataset(Dataset):
    def __init__(self, n_samples=100000, n_dim=3, method='lhs'):
        self.n_dim = n_dim
        self.n_samples = n_samples

        if method == 'sobol':
            engine = torch.quasirandom.SobolEngine(dimension=n_dim)
            samples = engine.draw(n_samples)
        elif method == 'lhs':
            samples = torch.tensor(lhs(n_dim, samples=n_samples), dtype=torch.float32)
        elif method == 'random':
            samples = torch.rand(n_samples, n_dim)
        else:
            raise ValueError("method must be 'sobol', 'lhs', or 'random'")

        # Map [0,1] → [-1,1]
        self.samples = 2 * samples - 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Each item is a 1D coordinate vector [t, y, x]
        return self.samples[idx]


class DataPointsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def NormalizeData(data):
    """
    Function which normalize data, and outputs its normalized version from -1 to 1
    
    Inputs:
        Tensor of any shape (In like reasonable format)

    Outputs:
        it also outputs lower and upper bound


    """ 
    data_max=data.max()
    data_min=data.min()

    normalized_data=2*(data-data_min)/(data_max-data_min)-1
    return normalized_data,data_max,data_min
