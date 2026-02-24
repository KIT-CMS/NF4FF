import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from CustomLogging import setup_logging
from typing import Any, List, Union
import numpy as np
logger = setup_logging(logger=logging.getLogger(__name__))



# ------ model ------


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        last = in_dim

        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h

        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Affine Coupling Layer
# -------------------------
class AffineCoupling(nn.Module):
    def __init__(self, dim, mask, hidden_dims=(128, 128), s_scale=2.0):
        super().__init__()

        self.dim = dim
        self.register_buffer("mask", mask)

        self.st_net = MLP(in_dim=dim, out_dim=2 * dim, hidden_dims=hidden_dims)
        self.s_scale = s_scale

    def forward(self, x):
        x_masked = x * self.mask

        s, t = torch.chunk(self.st_net(x_masked), 2, dim=-1)
        s = torch.tanh(s) * self.s_scale

        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = ((1 - self.mask) * s).sum(dim=-1)

        return y, log_det

    def inverse(self, y):
        y_masked = y * self.mask

        s, t = torch.chunk(self.st_net(y_masked), 2, dim=-1)
        s = torch.tanh(s) * self.s_scale

        x = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        return x



class RealNVP(nn.Module):
    """
    Stack of affine coupling layers with alternating masks.
    Base distribution: standard Normal.
    """
    def __init__(self, dim, n_layers=6, hidden_dims=(128, 128), s_scale=2.0, device= torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        torch.device(device)
        self.dim = dim
        logger.info(f"Dimension of RealNVP input: {self.dim}")
        base_mask = torch.tensor([i % 2 for i in range(dim)], dtype=torch.float32)

        masks = []
        for i in range(n_layers):
            if i % 2 == 0:
                masks.append(base_mask)
            else:
                masks.append(1 - base_mask)

        # create list of layers, 
        self.couplings = nn.ModuleList([
            AffineCoupling(dim=dim, mask=m, hidden_dims=hidden_dims, s_scale=s_scale)
            for m in masks
        ])

        # Learnable base distribution parameters (optional; here fixed to standard normal)
        self.register_buffer('base_mean', torch.zeros(dim))
        self.register_buffer('base_log_std', torch.zeros(dim))
                # StandardScaler or RobustScaler
        self.register_buffer("_scaler_shift", torch.full((dim,), 0.0))
        self.register_buffer("_scaler_scale", torch.full((dim,), 1.0))

    def _init_permute(self, m):
        # For invertible mixing layers, orthogonal init is robust
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
        elif isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_fn(self, module):
        # Generic global init (use cautiously; don't override the s,t zero head)
        if isinstance(module, nn.Linear) and module not in [self.permute]:
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)


    def f(self, x):
        """Forward through all couplings: x -> z. Returns z and sum log-dets."""
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.couplings:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def f_inv(self, z):
        """Inverse through all couplings: z -> x."""
        x = z
        # inverse in reverse order
        for layer in reversed(self.couplings):
            x = layer.inverse(x)
        return x

    def log_prob(self, x):
        """Compute log p(x) via change of variables."""
        z, log_det = self.f(x)
        # base log prob (diagonal Normal)
        std = torch.exp(self.base_log_std)
        log_pz = (-0.5 * (((z - self.base_mean) / std) ** 2).sum(dim=-1)
                  - 0.5 * self.dim * math.log(2 * math.pi)
                  - self.base_log_std.sum())
        return log_pz + log_det

    def sample(self, n):
        """Sample x by drawing z from base and mapping through inverse."""
        std = torch.exp(self.base_log_std)
        z = self.base_mean + std * torch.randn(n, self.dim, device=self.base_mean.device)
        x = self.f_inv(z)
        return x
    

    @property
    def _is_initialized(self):
        initialized = (torch.isnan(self._scaler_shift) | torch.isnan(self._scaler_scale)).sum() == 0
        initialized &= (self._scaler_scale != 1).all() & (self._scaler_shift != 0).all()
        return initialized

    def initialize_scaler(
        self,
        shift: Union[np.ndarray, torch.Tensor, None] = None,
        scale: Union[np.ndarray, torch.Tensor, None] = None,
    ):
        if self._is_initialized:
            logger.warning("Scaler already initialized. Overwriting the current values.")
        elif shift is not None and scale is not None:
            shift = torch.from_numpy(shift) if isinstance(shift, np.ndarray) else shift
            scale = torch.from_numpy(scale) if isinstance(scale, np.ndarray) else scale
        else:
            msg = """
                shift and scale must be both None
                (falling to default of shift=0.0, scale=1.0)
                or both not None raise ValueError
            """
            logger.error(msg)
            raise ValueError(msg)

        self._scaler_shift.data[:] = shift
        self._scaler_scale.data[:] = scale

    def apply_scaler(self, x):
        return (x - self._scaler_shift.to(x.device)) / self._scaler_scale.to(x.device)



    def forward(self, X):
        Xs = self.apply_scaler(X)
        return self.log_prob(Xs)
