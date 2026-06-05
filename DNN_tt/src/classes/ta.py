import itertools
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union, Literal

import dill
import more_itertools as mit
import numpy as np
import onnx
import torch as t

import CODE.LOGGING as log
import CODE.HELPER as helper

logger = log.setup_logging(logger=logging.getLogger(__name__))





@contextmanager
def eval_state(model: t.nn.Module) -> Iterable[t.nn.Module]:
    was_training = model.training
    try:
        model.eval()
        yield model
    finally:
        if was_training:
            model.train()


@contextmanager
def train_state(model: t.nn.Module) -> Iterable[t.nn.Module]:
    was_training = model.training
    try:
        model.train()
        yield model
    finally:
        if not was_training:
            model.eval()




def build_manual_scaler(model: t.nn.Module) -> Callable[[t.Tensor], t.Tensor]:

    if isinstance(model, FoldCombinedDNN):
        even_scaler = build_manual_scaler(model.even_model)
        odd_scaler = build_manual_scaler(model.odd_model)

        def _scaler_function(X: t.Tensor) -> t.Tensor:
            even_mask, features = (X[0, ...].long() % 2 == 0).squeeze(), X[1:, ...].T
            scaled_even, scaled_odd = even_scaler(features), odd_scaler(features)

            condition = even_mask.view(-1, 1) if scaled_even.dim() > 1 else even_mask
            scaled_features = t.where(condition, scaled_even, scaled_odd)

            out_X = X.clone()
            out_X[1:, ...] = scaled_features.T

            return out_X

        return _scaler_function

    elif isinstance(model, GroupedDNN):
        pipeline = [(conds, build_manual_scaler(payload)) for conds, payload in model._logic_pipeline]
        fallback_scaler = build_manual_scaler(model._fallback_payload)

        def _scaler_function(X: t.Tensor) -> t.Tensor:
            output, batch_size = t.zeros_like(X), X.shape[0]
            processed_mask = t.zeros(batch_size, dtype=t.bool, device=X.device)

            for conditions, payload_scaler in pipeline:
                current_mask = t.ones(batch_size, dtype=t.bool, device=X.device)
                for colume_idx, bounds in conditions:
                    vals = X[:, colume_idx]
                    if len(bounds) == 1:
                        current_mask = current_mask & (t.abs(vals - bounds[0]) < 1e-4)
                    elif len(bounds) == 2:
                        current_mask = current_mask & (vals >= bounds[0])
                        if bounds[1] != float("inf"):
                            current_mask = current_mask & (vals <= bounds[1])

                group_out, mask_float = payload_scaler(X), current_mask.to(dtype=X.dtype).unsqueeze(1)
                output = output + (group_out * mask_float)
                processed_mask = processed_mask | current_mask

            unprocessed_mask = ~processed_mask
            output = output + (fallback_scaler(X) * unprocessed_mask.to(dtype=X.dtype).unsqueeze(1))

            return output

        return _scaler_function

    elif hasattr(model, "wrapped_model"):
        return build_manual_scaler(model.wrapped_model)

    elif hasattr(model, "_scaler_shift") and hasattr(model, "_scaler_scale"):
        shift = model._scaler_shift.clone().detach()
        scale = model._scaler_scale.clone().detach()

        def _scaler_function(X: t.Tensor) -> t.Tensor:
            return (X - shift.to(X.device)) / scale.to(X.device)

        return _scaler_function

    else:
        raise ValueError(f"Cannot build manual scaler for type {type(model)}")


@contextmanager
def temporary_extract_scaler_callable(
    model: t.nn.Module,
) -> Iterable[Tuple[t.nn.Module, Callable[[t.Tensor], t.Tensor]]]:
    manual_scaler, originals = build_manual_scaler(model), {}

    try:
        for name, buffer in model.named_buffers():
            if name.endswith("_scaler_shift"):
                originals[name] = buffer.clone()
                buffer.fill_(0.0)
            elif name.endswith("_scaler_scale"):
                originals[name] = buffer.clone()
                buffer.fill_(1.0)

        yield model, manual_scaler

    finally:
        for name, buffer in model.named_buffers():
            if name in originals:
                buffer.copy_(originals[name])