import math
from collections.abc import Sequence
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from CustomLogging import setup_logging
from typing import Any, Callable, Dict, List, Union, Tuple
from abc import ABC, abstractmethod

import numpy as np
import more_itertools as mit
import itertools
import copy


logger = setup_logging(logger=logging.getLogger(__name__))


class CutAwarePreprocessingMixin:
    @staticmethod
    def _normalize_tail_values(value, cast_fn, name: str) -> List[Any]:
        if isinstance(value, (str, bytes)):
            return [cast_fn(value)]
        if isinstance(value, Sequence):
            values = [cast_fn(v) for v in value]
            if len(values) == 0:
                raise ValueError(f"{name} must not be empty.")
            return values
        return [cast_fn(value)]

    @staticmethod
    def _broadcast_tail_values(values, length: int, cast_fn, name: str) -> List[Any]:
        normalized = CutAwarePreprocessingMixin._normalize_tail_values(values, cast_fn, name)
        if len(normalized) == 1:
            return normalized * length
        if len(normalized) != length:
            raise ValueError(
                f"{name} must have length 1 or match the number of tail indices ({length})."
            )
        return normalized

    @staticmethod
    def _normalize_cut_indices(
        value: Union[int, Sequence[int], None],
        name: str,
    ) -> List[int]:
        if value is None:
            return []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = [int(v) for v in value]
        else:
            values = [int(value)]
        if len(values) == 0:
            raise ValueError(f"{name} must not be empty when provided.")
        return list(dict.fromkeys(values))

    @staticmethod
    def _normalize_cut_thresholds(
        value: Union[float, Sequence[float]],
        name: str,
    ) -> List[float]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = [float(v) for v in value]
        else:
            values = [float(value)]
        if len(values) == 0:
            raise ValueError(f"{name} must not be empty.")
        return values

    def _initialize_cut_preprocessing(
        self,
        dim: int,
        use_cut_preprocessing: bool = True,
        cut_preprocessing_index: Union[int, List[int], Tuple[int, ...], None] = None,
        cut_thresholds: Tuple[float, float] = (33.0, 30.0),
        cut_epsilon: float = 1e-6,
    ) -> None:
        self.register_buffer("_cut_preprocess_indices", t.empty(0, dtype=t.long))
        self.register_buffer("_cut_preprocess_thresholds", t.empty(0, dtype=t.float32))

        if not use_cut_preprocessing:
            return

        thresholds = self._normalize_cut_thresholds(cut_thresholds, "cut_preprocessing_thresholds")
        indices = self._normalize_cut_indices(cut_preprocessing_index, "cut_preprocessing_index")

        # Backward-compatible default: apply to the first N variables (historically pt_1 and pt_2).
        if len(indices) == 0:
            n_default = min(len(thresholds), dim)
            indices = list(range(n_default))
            thresholds = thresholds[:n_default]
        elif len(thresholds) == 1:
            thresholds = thresholds * len(indices)
        elif len(thresholds) != len(indices):
            raise ValueError(
                "cut_preprocessing_thresholds must have length 1 or match "
                "the number of cut_preprocessing_index entries."
            )

        for idx in indices:
            if not (0 <= idx < dim):
                raise ValueError(f"cut_preprocessing_index={idx} is out of bounds for dim={dim}.")

        self._cut_preprocess_epsilon = cut_epsilon
        self._cut_preprocess_indices = t.tensor(indices, dtype=t.long)
        self._cut_preprocess_thresholds = t.tensor(thresholds, dtype=t.float32)

    def apply_cut_preprocessing(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        if self._cut_preprocess_indices.numel() == 0:
            log_det = t.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            valid_mask = t.ones(x.shape[0], device=x.device, dtype=t.bool)
            return x, log_det, valid_mask

        indices = self._cut_preprocess_indices.to(x.device)
        thresholds = self._cut_preprocess_thresholds.to(x.device, x.dtype)
        transformed = x.clone()
        shifted = transformed[:, indices] - thresholds + self._cut_preprocess_epsilon
        valid_mask = (shifted > 0).all(dim=-1)
        safe_shifted = shifted.clamp_min(self._cut_preprocess_epsilon)
        transformed[:, indices] = t.log(safe_shifted)
        log_det = -t.log(safe_shifted).sum(dim=-1)
        return transformed, log_det, valid_mask

    def invert_cut_preprocessing(self, x: t.Tensor) -> t.Tensor:
        if self._cut_preprocess_indices.numel() == 0:
            return x

        indices = self._cut_preprocess_indices.to(x.device)
        thresholds = self._cut_preprocess_thresholds.to(x.device, x.dtype)
        restored = x.clone()
        restored[:, indices] = (
            thresholds + t.exp(restored[:, indices]) - self._cut_preprocess_epsilon
        )
        return restored

    def _initialize_tail_preprocessing(
        self,
        dim: int,
        use_tail_preprocessing: bool = False,
        tail_preprocessing_index: Union[int, List[int], Tuple[int, ...]] = 2,
        tail_preprocessing_type: Union[str, List[str], Tuple[str, ...]] = "asinh",
        tail_preprocessing_center: Union[float, List[float], Tuple[float, ...]] = 0.0,
        tail_preprocessing_scale: Union[float, List[float], Tuple[float, ...]] = 1.0,
        tail_preprocessing_epsilon: float = 1e-6,
    ) -> None:
        self._use_tail_preprocessing = bool(use_tail_preprocessing)
        self._tail_preprocessing_epsilon = float(tail_preprocessing_epsilon)
        self._tail_preprocessing_types: List[str] = []
        self.register_buffer("_tail_preprocessing_indices", t.empty(0, dtype=t.long))
        self.register_buffer("_tail_preprocessing_centers", t.empty(0, dtype=t.float32))
        self.register_buffer("_tail_preprocessing_scales", t.empty(0, dtype=t.float32))

        if self._use_tail_preprocessing:
            indices = self._normalize_tail_values(tail_preprocessing_index, int, "tail_preprocessing_index")
            indices = list(dict.fromkeys(indices))
            centers = self._broadcast_tail_values(tail_preprocessing_center, len(indices), float, "tail_preprocessing_center")
            scales = self._broadcast_tail_values(tail_preprocessing_scale, len(indices), float, "tail_preprocessing_scale")
            types = [value.lower() for value in self._broadcast_tail_values(tail_preprocessing_type, len(indices), str, "tail_preprocessing_type")]

            for idx in indices:
                if not (0 <= idx < dim):
                    raise ValueError(f"tail_preprocessing_index={idx} is out of bounds for dim={dim}.")
            for scale in scales:
                if scale <= 0:
                    raise ValueError("tail_preprocessing_scale must be > 0.")
            for tail_type in types:
                if tail_type not in {"asinh", "log1p"}:
                    raise ValueError("tail_preprocessing_type must be one of {'asinh', 'log1p'}.")

            self._tail_preprocessing_types = types
            self._tail_preprocessing_indices = t.tensor(indices, dtype=t.long)
            self._tail_preprocessing_centers = t.tensor(centers, dtype=t.float32)
            self._tail_preprocessing_scales = t.tensor(scales, dtype=t.float32)

    def apply_tail_preprocessing(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        log_det = t.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        valid_mask = t.ones(x.shape[0], device=x.device, dtype=t.bool)

        if not getattr(self, "_use_tail_preprocessing", False):
            return x, log_det, valid_mask

        eps = x.new_tensor(self._tail_preprocessing_epsilon)

        transformed = x.clone()

        indices = self._tail_preprocessing_indices.to(x.device)
        centers = self._tail_preprocessing_centers.to(x.device, x.dtype)
        scales = self._tail_preprocessing_scales.to(x.device, x.dtype)

        for pos, idx in enumerate(indices.tolist()):
            center = centers[pos]
            scale = scales[pos]
            u = (transformed[:, idx] - center) / scale

            if self._tail_preprocessing_types[pos] == "asinh":
                transformed[:, idx] = t.asinh(u)
                log_det = log_det - t.log(scale) - 0.5 * t.log1p(u * u)
                continue

            valid_mask_i = (1.0 + u) > 0.0
            valid_mask = valid_mask & valid_mask_i
            safe_u = t.where(valid_mask_i, u, (-1.0 + eps).to(x.dtype))
            transformed[:, idx] = t.log1p(safe_u)
            log_det = log_det - t.log(scale) - t.log1p(safe_u)

        return transformed, log_det, valid_mask

    def invert_tail_preprocessing(self, x: t.Tensor) -> t.Tensor:
        if not getattr(self, "_use_tail_preprocessing", False):
            return x

        restored = x.clone()

        indices = self._tail_preprocessing_indices.to(x.device)
        centers = self._tail_preprocessing_centers.to(x.device, x.dtype)
        scales = self._tail_preprocessing_scales.to(x.device, x.dtype)

        for pos, idx in enumerate(indices.tolist()):
            center = centers[pos]
            scale = scales[pos]
            if self._tail_preprocessing_types[pos] == "asinh":
                restored[:, idx] = center + scale * t.sinh(restored[:, idx])
            else:
                restored[:, idx] = center + scale * t.expm1(restored[:, idx])

        return restored

    def apply_preprocessing(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        x_cut, log_det_cut, valid_cut = self.apply_cut_preprocessing(x)
        x_tail, log_det_tail, valid_tail = self.apply_tail_preprocessing(x_cut)
        return x_tail, (log_det_cut + log_det_tail), (valid_cut & valid_tail)

    def invert_preprocessing(self, x: t.Tensor) -> t.Tensor:
        return self.invert_cut_preprocessing(self.invert_tail_preprocessing(x))


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


class AffineCoupling(nn.Module):
    def __init__(self, dim, mask, hidden_dims=(128, 128), s_scale=2.0):
        super().__init__()

        self.dim = dim
        self.register_buffer("mask", mask)

        self.st_net = MLP(in_dim=dim, out_dim=2 * dim, hidden_dims=hidden_dims)
        self.s_scale = s_scale

    def forward(self, x):
        x_masked = x * self.mask

        s, shift = t.chunk(self.st_net(x_masked), 2, dim=-1)
        s = t.tanh(s) * self.s_scale

        y = x_masked + (1 - self.mask) * (x * t.exp(s) + shift)
        log_det = ((1 - self.mask) * s).sum(dim=-1)

        return y, log_det

    def inverse(self, y):
        y_masked = y * self.mask

        s, shift = t.chunk(self.st_net(y_masked), 2, dim=-1)
        s = t.tanh(s) * self.s_scale

        x = y_masked + (1 - self.mask) * ((y - shift) * t.exp(-s))
        return x


class RealNVP(CutAwarePreprocessingMixin, nn.Module):
    def __init__(self, dim, n_layers=6, hidden_dims=(128, 128), s_scale=2.0, device= t.device("cuda" if t.cuda.is_available() else "cpu"), use_cut_preprocessing=True, cut_preprocessing_index=None, cut_preprocessing_thresholds=(33.0, 30.0), cut_preprocessing_epsilon=1e-6, use_tail_preprocessing=False, tail_preprocessing_index=2, tail_preprocessing_type="asinh", tail_preprocessing_center=0.0, tail_preprocessing_scale=1.0, tail_preprocessing_epsilon=1e-6):
        super().__init__()
        t.device(device)
        self.dim = dim
        logger.info(f"Dimension of RealNVP input: {self.dim}")
        self._initialize_cut_preprocessing(
            dim=self.dim,
            use_cut_preprocessing=use_cut_preprocessing,
            cut_preprocessing_index=cut_preprocessing_index,
            cut_thresholds=cut_preprocessing_thresholds,
            cut_epsilon=cut_preprocessing_epsilon,
        )
        self._initialize_tail_preprocessing(
            dim=self.dim,
            use_tail_preprocessing=use_tail_preprocessing,
            tail_preprocessing_index=tail_preprocessing_index,
            tail_preprocessing_type=tail_preprocessing_type,
            tail_preprocessing_center=tail_preprocessing_center,
            tail_preprocessing_scale=tail_preprocessing_scale,
            tail_preprocessing_epsilon=tail_preprocessing_epsilon,
        )
        base_mask = t.tensor([i % 2 for i in range(dim)], dtype=t.float32)

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
        self.register_buffer('base_mean', t.zeros(dim))
        self.register_buffer('base_log_std', t.zeros(dim))
                # StandardScaler or RobustScaler
        self.register_buffer("_scaler_shift", t.full((dim,), 0.0))
        self.register_buffer("_scaler_scale", t.full((dim,), 1.0))

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
        log_det_total = t.zeros(x.shape[0], device=x.device)
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
        std = t.exp(self.base_log_std)
        log_pz = (-0.5 * (((z - self.base_mean) / std) ** 2).sum(dim=-1)
                  - 0.5 * self.dim * math.log(2 * math.pi)
                  - self.base_log_std.sum())
        return log_pz + log_det

    def sample(self, n):
        """Sample x by drawing z from base and mapping through inverse."""
        std = t.exp(self.base_log_std)
        z = self.base_mean + std * t.randn(n, self.dim, device=self.base_mean.device)
        x_scaled = self.f_inv(z)
        # map back to original (un-scaled) space: x = x_scaled * scale + shift
        x_preprocessed = x_scaled * self._scaler_scale.to(x_scaled.device) + self._scaler_shift.to(x_scaled.device)
        x = self.invert_preprocessing(x_preprocessed)
        return x
    

    @property
    def _is_initialized(self):
        initialized = (t.isnan(self._scaler_shift) | t.isnan(self._scaler_scale)).sum() == 0
        initialized &= (self._scaler_scale != 1).all() & (self._scaler_shift != 0).all()
        return initialized

    def initialize_scaler(
        self,
        shift: Union[np.ndarray, t.Tensor, None] = None,
        scale: Union[np.ndarray, t.Tensor, None] = None,
    ):
        # Both shift and scale must be provided (no partial initialization)
        if shift is None or scale is None:
            msg = (
                "shift and scale must be both provided (not None)."
                " To reset to defaults explicitly set shift=0 and scale=1."
            )
            logger.error(msg)
            raise ValueError(msg)

        # convert numpy arrays to tensors if needed
        shift = t.from_numpy(shift) if isinstance(shift, np.ndarray) else shift
        scale = t.from_numpy(scale) if isinstance(scale, np.ndarray) else scale

        if self._is_initialized:
            logger.warning("Scaler already initialized. Overwriting the current values.")

        self._scaler_shift.data[:] = shift
        self._scaler_scale.data[:] = scale

    def apply_scaler(self, x):
        return (x - self._scaler_shift.to(x.device)) / self._scaler_scale.to(x.device)



    def forward(self, X):
        X_preprocessed, log_det_preprocess, valid_mask = self.apply_preprocessing(X)
        Xs = self.apply_scaler(X_preprocessed)
        # log prob of the scaled input
        logp_scaled = self.log_prob(Xs)
        # account for the scaling Jacobian: Xs = (X - shift)/scale -> det = prod(1/scale)
        # so log|det| = -sum(log(scale)). Add this constant per-sample.
        log_det_scale = -t.log(self._scaler_scale.to(X.device)).sum()
        total_log_prob = logp_scaled + log_det_scale + log_det_preprocess
        invalid_log_prob = t.full_like(total_log_prob, -t.inf)
        return t.where(valid_mask, total_log_prob, invalid_log_prob)


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, dim, cond_dim, mask, hidden_dims=(128, 128), s_scale=2.0):
        super().__init__()

        self.dim = dim
        self.cond_dim = cond_dim
        self.register_buffer("mask", mask)

        self.st_net = MLP(in_dim=dim + cond_dim, out_dim=2 * dim, hidden_dims=hidden_dims)
        self.s_scale = s_scale

    def forward(self, x, cond):
        if cond.dim() == 1:
            cond = cond.unsqueeze(-1)

        x_masked = x * self.mask
        st_input = t.cat([x_masked, cond], dim=-1)

        s, shift = t.chunk(self.st_net(st_input), 2, dim=-1)
        s = t.tanh(s) * self.s_scale

        y = x_masked + (1 - self.mask) * (x * t.exp(s) + shift)
        log_det = ((1 - self.mask) * s).sum(dim=-1)

        return y, log_det

    def inverse(self, y, cond):
        if cond.dim() == 1:
            cond = cond.unsqueeze(-1)

        y_masked = y * self.mask
        st_input = t.cat([y_masked, cond], dim=-1)

        s, shift = t.chunk(self.st_net(st_input), 2, dim=-1)
        s = t.tanh(s) * self.s_scale

        x = y_masked + (1 - self.mask) * ((y - shift) * t.exp(-s))
        return x


class ConditionalRealNVP(CutAwarePreprocessingMixin, nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=1,
        n_layers=6,
        hidden_dims=(128, 128),
        s_scale=2.0,
        device=t.device("cuda" if t.cuda.is_available() else "cpu"),
        use_cut_preprocessing=True,
        cut_preprocessing_index=None,
        cut_preprocessing_thresholds=(33.0, 30.0),
        cut_preprocessing_epsilon=1e-6,
        use_tail_preprocessing=False,
        tail_preprocessing_index=2,
        tail_preprocessing_type="asinh",
        tail_preprocessing_center=0.0,
        tail_preprocessing_scale=1.0,
        tail_preprocessing_epsilon=1e-6,
    ):
        super().__init__()
        t.device(device)

        self.dim = dim
        self.cond_dim = cond_dim
        logger.info(f"Dimension of ConditionalRealNVP feature input: {self.dim}, cond_dim: {self.cond_dim}")
        self._initialize_cut_preprocessing(
            dim=self.dim,
            use_cut_preprocessing=use_cut_preprocessing,
            cut_preprocessing_index=cut_preprocessing_index,
            cut_thresholds=cut_preprocessing_thresholds,
            cut_epsilon=cut_preprocessing_epsilon,
        )
        self._initialize_tail_preprocessing(
            dim=self.dim,
            use_tail_preprocessing=use_tail_preprocessing,
            tail_preprocessing_index=tail_preprocessing_index,
            tail_preprocessing_type=tail_preprocessing_type,
            tail_preprocessing_center=tail_preprocessing_center,
            tail_preprocessing_scale=tail_preprocessing_scale,
            tail_preprocessing_epsilon=tail_preprocessing_epsilon,
        )

        base_mask = t.tensor([i % 2 for i in range(dim)], dtype=t.float32)

        masks = []
        for i in range(n_layers):
            if i % 2 == 0:
                masks.append(base_mask)
            else:
                masks.append(1 - base_mask)

        self.couplings = nn.ModuleList([
            ConditionalAffineCoupling(
                dim=dim,
                cond_dim=cond_dim,
                mask=m,
                hidden_dims=hidden_dims,
                s_scale=s_scale,
            )
            for m in masks
        ])

        self.register_buffer('base_mean', t.zeros(dim))
        self.register_buffer('base_log_std', t.zeros(dim))

        self.register_buffer("_scaler_shift", t.full((dim,), 0.0))
        self.register_buffer("_scaler_scale", t.full((dim,), 1.0))

    def f(self, x, cond):
        log_det_total = t.zeros(x.shape[0], device=x.device)
        z = x

        for layer in self.couplings:
            z, log_det = layer(z, cond)
            log_det_total = log_det_total + log_det

        return z, log_det_total

    def f_inv(self, z, cond):
        x = z

        for layer in reversed(self.couplings):
            x = layer.inverse(x, cond)

        return x

    def log_prob(self, x, cond):
        z, log_det = self.f(x, cond)

        std = t.exp(self.base_log_std)
        log_pz = (
            -0.5 * (((z - self.base_mean) / std) ** 2).sum(dim=-1)
            - 0.5 * self.dim * math.log(2 * math.pi)
            - self.base_log_std.sum()
        )

        return log_pz + log_det

    def sample(self, cond):
        if cond.dim() == 1:
            cond = cond.unsqueeze(-1)

        n = cond.shape[0]
        std = t.exp(self.base_log_std)
        z = self.base_mean + std * t.randn(n, self.dim, device=self.base_mean.device)

        cond = cond.to(self.base_mean.device)
        x_scaled = self.f_inv(z, cond)
        x_preprocessed = x_scaled * self._scaler_scale.to(x_scaled.device) + self._scaler_shift.to(x_scaled.device)
        x = self.invert_preprocessing(x_preprocessed)

        return x

    @property
    def _is_initialized(self):
        initialized = (t.isnan(self._scaler_shift) | t.isnan(self._scaler_scale)).sum() == 0
        initialized &= (self._scaler_scale != 1).all() & (self._scaler_shift != 0).all()
        return initialized

    def initialize_scaler(
        self,
        shift: Union[np.ndarray, t.Tensor, None] = None,
        scale: Union[np.ndarray, t.Tensor, None] = None,
    ):
        if shift is None or scale is None:
            msg = (
                "shift and scale must be both provided (not None)."
                " To reset to defaults explicitly set shift=0 and scale=1."
            )
            logger.error(msg)
            raise ValueError(msg)

        shift = t.from_numpy(shift) if isinstance(shift, np.ndarray) else shift
        scale = t.from_numpy(scale) if isinstance(scale, np.ndarray) else scale

        if self._is_initialized:
            logger.warning("Scaler already initialized. Overwriting the current values.")

        self._scaler_shift.data[:] = shift
        self._scaler_scale.data[:] = scale

    def apply_scaler(self, x):
        return (x - self._scaler_shift.to(x.device)) / self._scaler_scale.to(x.device)

    def forward(self, X):
        cond = X[:, :self.cond_dim]
        x = X[:, self.cond_dim:]

        x_preprocessed, log_det_preprocess, valid_mask = self.apply_preprocessing(x)
        Xs = self.apply_scaler(x_preprocessed)
        logp_scaled = self.log_prob(Xs, cond)

        log_det_scale = -t.log(self._scaler_scale.to(X.device)).sum()
        total_log_prob = logp_scaled + log_det_scale + log_det_preprocess
        invalid_log_prob = t.full_like(total_log_prob, -t.inf)
        return t.where(valid_mask, total_log_prob, invalid_log_prob)


class RealNVP_NN(CutAwarePreprocessingMixin, t.nn.Module):
    def __init__(
        self,
        input_nodes: int,
        hidden_nodes: Tuple[int, ...] = (128, 128),
        n_layers: int = 6,
        dropout: Union[float, Tuple[float, ...]] = 0.0,
        activation: Union[Callable, str] = "ReLU",
        s_scale: float = 2.0,
        input_names: Union[List[str], None] = None,
        use_cut_preprocessing: bool = True,
        cut_preprocessing_index: Union[int, List[int], Tuple[int, ...], None] = None,
        cut_preprocessing_thresholds: Tuple[float, float] = (33.0, 30.0),
        cut_preprocessing_epsilon: float = 1e-6,
        use_tail_preprocessing: bool = False,
        tail_preprocessing_index: int = 2,
        tail_preprocessing_type: str = "asinh",
        tail_preprocessing_center: float = 0.0,
        tail_preprocessing_scale: float = 1.0,
        tail_preprocessing_epsilon: float = 1e-6,
    ):
        super().__init__()

        self._input_nodes = input_nodes
        self._hidden_nodes = hidden_nodes
        self._n_layers = n_layers
        self._dropout = dropout if isinstance(dropout, tuple) else (dropout,) * len(hidden_nodes)
        self._activation = activation
        self._s_scale = s_scale
        self._input_names = input_names
        self._output_nodes = 1
        self._output_activation = None

        self._forward_auto_to_device = True

        self.dim = input_nodes
        self._initialize_cut_preprocessing(
            dim=self.dim,
            use_cut_preprocessing=use_cut_preprocessing,
            cut_preprocessing_index=cut_preprocessing_index,
            cut_thresholds=cut_preprocessing_thresholds,
            cut_epsilon=cut_preprocessing_epsilon,
        )
        self._initialize_tail_preprocessing(
            dim=self.dim,
            use_tail_preprocessing=use_tail_preprocessing,
            tail_preprocessing_index=tail_preprocessing_index,
            tail_preprocessing_type=tail_preprocessing_type,
            tail_preprocessing_center=tail_preprocessing_center,
            tail_preprocessing_scale=tail_preprocessing_scale,
            tail_preprocessing_epsilon=tail_preprocessing_epsilon,
        )

        if isinstance(activation, str) and hasattr(t.nn, activation):
            activation_layer = getattr(t.nn, activation)()
        elif callable(activation):
            activation_layer = activation
        else:
            raise TypeError("Not a valid activation function")

        self._activation_layer = activation_layer

        base_mask = t.tensor([i % 2 for i in range(self.dim)], dtype=t.float32)

        masks = []
        for i in range(n_layers):
            masks.append(base_mask if i % 2 == 0 else 1 - base_mask)

        self.register_buffer("masks", t.stack(masks))

        # coupling networks
        self.st_nets = t.nn.ModuleList([
            self._build_st_net() for _ in range(n_layers)
        ])

        # base distribution
        self.register_buffer("base_mean", t.zeros(self.dim))
        self.register_buffer("base_log_std", t.zeros(self.dim))

        # scaler
        self.register_buffer("_scaler_shift", t.full((input_nodes,), 0.0))
        self.register_buffer("_scaler_scale", t.full((input_nodes,), 1.0))


    def _build_st_net(self):

        layers = []

        nodes = [self.dim] + list(self._hidden_nodes)

        for (n1, n2), drop in zip(
            mit.pairwise(nodes),
            [0.0] + list(self._dropout)
        ):
            layers.extend([
                t.nn.Linear(n1, n2),
                self._activation_layer,
                t.nn.Dropout(drop),
            ])

        layers.append(t.nn.Linear(self._hidden_nodes[-1], 2 * self.dim))

        return t.nn.Sequential(*layers)

    def f(self, x):

        log_det_total = t.zeros(x.shape[0], device=x.device)

        z = x

        for mask, st_net in zip(self.masks, self.st_nets):

            mask = mask.to(z.device)

            z_masked = z * mask

            s, t_shift = t.chunk(st_net(z_masked), 2, dim=-1)

            s = t.tanh(s) * self._s_scale

            z = z_masked + (1 - mask) * (z * t.exp(s) + t_shift)

            log_det = ((1 - mask) * s).sum(dim=-1)

            log_det_total += log_det

        return z, log_det_total

    def f_inv(self, z):

        x = z

        for mask, st_net in reversed(list(zip(self.masks, self.st_nets))):

            mask = mask.to(x.device)

            x_masked = x * mask

            s, t_shift = t.chunk(st_net(x_masked), 2, dim=-1)

            s = t.tanh(s) * self._s_scale

            x = x_masked + (1 - mask) * ((x - t_shift) * t.exp(-s))

        return x

    def log_prob(self, x):

        z, log_det = self.f(x)

        std = t.exp(self.base_log_std)

        log_pz = (
            -0.5 * (((z - self.base_mean) / std) ** 2).sum(dim=-1)
            - 0.5 * self.dim * math.log(2 * math.pi)
            - self.base_log_std.sum()
        )

        return log_pz + log_det

    def sample(self, n):

        std = t.exp(self.base_log_std)

        z = self.base_mean + std * t.randn(n, self.dim, device=self.base_mean.device)

        x_scaled = self.f_inv(z)

        x = (
            x_scaled * self._scaler_scale.to(x_scaled.device)
            + self._scaler_shift.to(x_scaled.device)
        )

        x = self.invert_preprocessing(x)

        return x


    @property
    def _is_initialized(self):

        initialized = (t.isnan(self._scaler_shift) | t.isnan(self._scaler_scale)).sum() == 0
        initialized &= (self._scaler_scale != 1).all() & (self._scaler_shift != 0).all()

        return initialized

    def initialize_scaler(
        self,
        shift: Union[np.ndarray, t.Tensor, None] = None,
        scale: Union[np.ndarray, t.Tensor, None] = None,
        safety_epsilon: float = 1e-6,
    ):

        if shift is None or scale is None:
            raise ValueError("shift and scale must both be provided")

        shift = t.from_numpy(shift) if isinstance(shift, np.ndarray) else shift
        scale = t.from_numpy(scale) if isinstance(scale, np.ndarray) else scale

        if safety_epsilon is not None:
            scale = scale.clamp(min=safety_epsilon)

        if (scale == 0).any():
            raise ValueError("Scaler scale contains zeros")

        self._scaler_shift.data[:] = shift
        self._scaler_scale.data[:] = scale

    def apply_scaler(self, x):

        return (x - self._scaler_shift.to(x.device)) / self._scaler_scale.to(x.device)



    def forward(self, X):

        if self._forward_auto_to_device:
            X = X.to(self.base_mean.device)

        X_preprocessed, log_det_preprocess, valid_mask = self.apply_preprocessing(X)

        Xs = self.apply_scaler(X_preprocessed)

        logp_scaled = self.log_prob(Xs)

        log_det_scale = -t.log(self._scaler_scale.to(X.device)).sum()

        total_log_prob = logp_scaled + log_det_scale + log_det_preprocess
        invalid_log_prob = t.full_like(total_log_prob, -t.inf)

        return t.where(valid_mask, total_log_prob, invalid_log_prob)


class GroupedLayerABC(t.nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self._logic_pipeline: List[Tuple[Any, Any]] = []
        self._fallback_payload: Any = None
        self._wrapped_delegate: Any = None

    @abstractmethod
    def _execute_group(self, X: t.Tensor, payload: Any) -> t.Tensor:
        pass

    def forward(self, X: t.Tensor) -> t.Tensor:
        fallback_out = self._execute_group(X, self._fallback_payload)

        if fallback_out.dim() == 1:
            fallback_out = fallback_out.unsqueeze(1)

        if not self._logic_pipeline:
            return fallback_out

        batch_size = X.shape[0]

        output = t.zeros_like(fallback_out)
        processed_mask = t.zeros(batch_size, dtype=t.bool, device=X.device)

        for conditions, payload in self._logic_pipeline:
            current_mask = t.ones(batch_size, dtype=t.bool, device=X.device)
            for colume_idx, bounds in conditions:
                vals = X[:, colume_idx]

                if len(bounds) == 1:  # checks at trace time, not ONNX run time
                    current_mask = current_mask & (vals == bounds[0])
                elif len(bounds) == 2:
                    lower, upper = bounds
                    current_mask = current_mask & (vals >= lower)
                    if upper != float("inf"):
                        current_mask = current_mask & (vals <= upper)
                else:
                    raise ValueError(f"Invalid bound: {bounds}")

            group_out = self._execute_group(X, payload)

            if group_out.dim() == 1:
                group_out = group_out.unsqueeze(1)

            mask_float = current_mask.to(dtype=output.dtype).unsqueeze(1)

            output = output + (group_out * mask_float)
            processed_mask = processed_mask | current_mask

        unprocessed_mask = ~processed_mask
        unprocessed_mask_float = unprocessed_mask.to(dtype=output.dtype).unsqueeze(1)
        output = output + (fallback_out * unprocessed_mask_float)

        return output

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if self._wrapped_delegate is not None:
                return getattr(self._wrapped_delegate, name)
            raise

    @property
    def _imports(self) -> str:
        imports = set()

        if self.__class__.__module__ != "builtins":
            imports.add(f"from {self.__class__.__module__} import {self.__class__.__name__}")

        imports.add("from math import inf")

        def extract_recursive(obj: Any):
            if hasattr(obj, "__class__"):
                cls = obj.__class__
                if cls.__module__ != "builtins":
                    imports.add(f"from {cls.__module__} import {cls.__name__}")

            if hasattr(obj, "_imports"):
                for line in obj._imports.split("\n"):
                    if line.strip():
                        imports.add(line)

        extract_recursive(self._fallback_payload)

        for _, payload in self._logic_pipeline:
            extract_recursive(payload)

        if self._wrapped_delegate is not None:
            extract_recursive(self._wrapped_delegate)

        return "\n".join(sorted(list(imports))) + "\n"

    def __recreate__(self) -> str:
        return f"{self._imports}__model = {self.model_name}\n\n"


class GroupedNFRouter(GroupedLayerABC):
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList()  # ADD THIS

    # ---- mask for clipping and nonzero PDFs

    def _execute_group(self, X, payload):
        X_nf = X[:, 1:]
        return payload(X_nf)
    
    @property
    def model_name(self):
        return "GroupedNFRouter"
    

    # ----- model -----


class BinaryClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 200,
        p: float = 0.1,
        hidden_layers: int = 2,
    ):
        super().__init__()
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be >= 0")

        layers = []
        in_features = input_dim
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=p),
                ]
            )
            in_features = hidden_dim

        layers.extend(
            [
                nn.Linear(in_features, 1),
                nn.Sigmoid(),
            ]
        )

        self.net = nn.Sequential(*layers)

        # registered buffers (no-op because we pre-scale; left for compatibility)
        self.register_buffer("_scaler_shift", t.full((input_dim,), 0.0, dtype=t.float32))
        self.register_buffer("_scaler_scale", t.full((input_dim,), 1.0, dtype=t.float32))

    @property
    def _is_initialized(self):
        initialized = (t.isnan(self._scaler_shift) | t.isnan(self._scaler_scale)).sum() == 0
        initialized &= (self._scaler_scale != 1).all() & (self._scaler_shift != 0).all()
        return initialized

    def initialize_scaler(
        self,
        shift: Union[np.ndarray, t.Tensor, None] = None,
        scale: Union[np.ndarray, t.Tensor, None] = None,
    ):
        if self._is_initialized:
            logger.warning("Scaler already initialized. Overwriting the current values.")
        elif shift is not None and scale is not None:
            shift = t.from_numpy(shift) if isinstance(shift, np.ndarray) else shift
            scale = t.from_numpy(scale) if isinstance(scale, np.ndarray) else scale
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

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.apply_scaler(x)
        return self.net(x)

