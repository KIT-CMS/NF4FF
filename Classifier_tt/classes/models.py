import torch
import torch as t
import torch.nn as nn
import numpy as np
import logging
from typing import Union
from classes.Logging import setup_logging

# ------ logger -----

logger = setup_logging(logger=logging.getLogger(__name__))

# ----- model -----
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 200, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=p),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=p),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # registered buffers (no-op because we pre-scale; left for compatibility)
        self.register_buffer("_scaler_shift", torch.full((input_dim,), 0.0, dtype=torch.float32))
        self.register_buffer("_scaler_scale", torch.full((input_dim,), 1.0, dtype=torch.float32))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.apply_scaler(x)
        return self.net(x)

class FoldCombinedDNN_(t.nn.Module):
    def __init__(self, even_model: BinaryClassifier, odd_model: BinaryClassifier) -> None:
        super(FoldCombinedDNN_, self).__init__()
        self.even_model = even_model  # Even Model: Trained on ODD -> Use for EVEN events
        self.odd_model = odd_model  # Odd Model: Trained on EVEN -> Use for ODD events
        self._input_nodes = even_model._input_nodes + 1  # +1 for event ID

    def forward(self, x: t.Tensor) -> t.Tensor:
        even_event_ids, features = (x[..., 0].long() % 2 == 0).squeeze(), x[..., 1:]
        return t.where(even_event_ids, self.even_model(features).squeeze(), self.odd_model(features).squeeze())
