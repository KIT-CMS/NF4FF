from __future__ import annotations
from typing import Optional, Callable, Union, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

ArrayLike = Union[np.ndarray, torch.Tensor]

class Datasets():
    SR_like_tight = pd.read_pickle('../data/SR_like_tight.pkl')
    SR_like_loose = pd.read_pickle('../data/SR_like_loose.pkl')
    SR_tight = pd.read_pickle('../data/SR_tight.pkl')
    SR_loose = pd.read_pickle('../data/SR_loose.pkl')

    AR_like_tight = pd.read_pickle('../data/AR_like_tight.pkl')
    AR_like_loose = pd.read_pickle('../data/AR_like_loose.pkl')
    AR_tight = pd.read_pickle('../data/AR_tight.pkl')
    AR_loose = pd.read_pickle('../data/AR_loose.pkl')



class weightedDataset(Dataset):
    def __init__(self, X, w, dtype_features = torch.float32):
        assert len(X) == len(w), "X and w must have same length."
        self.X = torch.as_tensor(X, dtype=dtype_features)
        self.w = torch.as_tensor(w, dtype = torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.w[idx]

'''
class NDArrayDataset(Dataset):
    """
    A simple dataset for n-dimensional numeric data (X[, y]).
    - X: shape (N, D) or any per-sample shape (e.g., (T, F))
    - y: optional labels/targets, shape (N,) or (N, K)
    - transform: optional callable applied to X (or to (X, y) if it expects two args)
    """
    def __init__(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        transform: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
        y_dtype: Optional[torch.dtype] = None,
    ):
        # Convert NumPy to Torch
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if y is not None and isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        # Dtypes
        self.X = X.to(dtype)
        self.y = None if y is None else (y.to(y_dtype) if y_dtype is not None else y)
        self.transform = transform

        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError(f"X and y must have the same length, got {len(self.X)} vs {len(self.y)}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.y is None:
            return self._apply_transform(x)
        else:
            y = self.y[idx]
            tx = self._apply_transform(x, y)
            # Allow transform to return either x or (x, y)
            if isinstance(tx, tuple) and len(tx) == 2:
                return tx
            return tx, y

    def _apply_transform(self, x, y=None):
        if self.transform is None:
            return x if y is None else (x, y)
        # Support transforms that accept (x,y) or x only
        try:
            return self.transform(x, y) if y is not None else self.transform(x)
        except TypeError:
            return self.transform(x)

    # Convenience
    def get_all_X(self) -> torch.Tensor:
        return self.X
'''