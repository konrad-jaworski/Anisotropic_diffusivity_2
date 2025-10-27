import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import qmc



def NormalizeData(data):
    """
    Function which normalize data, by its Max absolute value
    
    Inputs:
        torch:Tensor

    Outputs:
        torch:Normalized tensor
        scalar: Normalizing factor
    """ 
    data_max=data.max()
    data_min=data.min()

    if abs(data_max)>abs(data_min):
        scaling_factor=abs(data_max)
    else:
        scaling_factor=abs(data_min)

    scaled_tensor=data/scaling_factor
    return scaled_tensor,scaling_factor 


class DomainDataset(Dataset):
    """
    Provides collocation points for PDE loss function
    """
    def __init__(self, n_samples=100000, n_dim=3, method='lhs'):
        self.n_dim = n_dim
        self.n_samples = n_samples

        if method == 'sobol':
            engine = torch.quasirandom.SobolEngine(dimension=n_dim)
            self.samples = engine.draw(n_samples)
        elif method == 'lhs':
            sampler = qmc.LatinHypercube(d=n_dim)
            samples = sampler.random(n=n_samples)
            self.samples = torch.tensor(samples, dtype=torch.float32)
        elif method == 'random':
            self.samples = torch.rand(n_samples, n_dim)
        else:
            raise ValueError("method must be 'sobol', 'lhs', or 'random'")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.samples[idx]   

class DataPointsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


