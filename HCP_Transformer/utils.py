import numpy as np
import torch
from torch.utils.data import Dataset

def bin_y(y, num_bins=5):
    # scale the y data
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    # bin the y data
    bins = [
        (0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.0)]

    y_binned = np.zeros(y.shape)
    for i in range(num_bins):
        y_binned[(y >= bins[i][0]) & (y < bins[i][1])] = i

    return y_binned

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std[self.std == 0] = 1e-8

    def transform(self, x):
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self, x):
        return (x * self.std) + self.mean
    

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]