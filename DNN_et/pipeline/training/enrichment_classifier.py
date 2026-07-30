"""Enrichment classifier training primitives."""

import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import random
from torch.utils import data
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from logging_utils.context import setup_logging
from core.config import load_config
from data.components import (
    CollectionMeta,
    get_class_weights,
    _same_sign_opposite_sign_split,
    _collection,
)
import torch as t
from tap import Tap
from typing import Any, Callable, Dict, Generator, List, Literal, Tuple, Union
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass
import time
from rich.console import Console
from rich.table import Table

console = Console()

logger = setup_logging(logger=logging.getLogger(__name__))



QCD_WEIGHT_BINNING = 'quantile'
QCD_WEIGHT_N_BINS = 40
QCD_WEIGHT_DYNAMIC_DELTA = 10.0
QCD_WEIGHT_DYNAMIC_DELTA_LAST = 10.0
QCD_WEIGHT_DYNAMIC_MIN_QCD_YIELD = 10.0
QCD_WEIGHT_REFRESH_EVERY = 5
QCD_WEIGHT_REFRESH_UNTIL_EPOCH = 100
QCD_SS_WEIGHT_DYNAMIC_DELTA = 10.0
QCD_SS_WEIGHT_DYNAMIC_DELTA_LAST = 10.0
QCD_SS_WEIGHT_DYNAMIC_MIN_QCD_YIELD = 10.0

# ----- data clas
@dataclass
class _component_collection(metaclass=CollectionMeta):
    _: KW_ONLY
    X: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    Y: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    weights: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    class_weights: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    process: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    qcd_weights_os: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    qcd_weights_ss: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    SR_like: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    parity: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    row_index: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None

def get_my_data(df, training_var):

    ss_region = df[df.SS]
    os_region = df[(df.OS & (df.Label != 2)) | (df.SS & (df.Label == 2))]

    ss_os_split = _same_sign_opposite_sign_split(
        ss = ss_region,
        os = os_region
    )

    return _component_collection(
        X = ss_os_split.apply_func(lambda x: x[training_var].to_numpy(dtype=np.float32)),
        Y = ss_os_split.apply_func(lambda x: x["Label"].to_numpy(dtype=np.float32)),
        weights = ss_os_split.apply_func(lambda x: x["weight"].to_numpy(dtype=np.float32)),
        class_weights = ss_os_split.apply_func(lambda x: x["class_weights"].to_numpy()),
        process = ss_os_split.apply_func(lambda x: x["process"].to_numpy(dtype=np.float32)),
        SR_like = ss_os_split.apply_func(lambda x: x["id_tau_vsJet_Tight_2"].to_numpy(dtype=np.float32)),
        parity = ss_os_split.apply_func(lambda x: (x["event"] % 2).to_numpy(dtype=np.int64)),
        row_index = ss_os_split.apply_func(lambda x: x.index.to_numpy(dtype=np.int64)),
    )

@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    X = X.to(device, non_blocking=True)
    logits = model(X)
    return logits.squeeze(1).cpu()

def set_negatives_to_one(tensor):
    # Using torch.where
    return torch.where(tensor < 0, torch.ones_like(tensor), tensor)

def get_class_weights(
    weights: t.Tensor,
    Y: t.Tensor,
    classes: tuple = (0, 1),
    class_weighted: bool = True,
) -> t.Tensor:

    weights = weights.float()
    Y = Y.long()

    _weights = torch.zeros_like(weights)

    total_weight = weights.sum()

    for _class in classes:
        mask = (Y == _class)
        class_sum = weights[mask].sum()

        if class_sum > 0:
            _weights[mask] = total_weight / class_sum
        else:
            _weights[mask] = 0.0

    return _weights * (weights if class_weighted else 1.0)


def should_refresh_qcd_weights(epoch: int) -> bool:
    return True



def refresh_qcd_weights(
    dataset: _component_collection,
    model: nn.Module,
    qcd_mask_os_loaded: t.Tensor,
    device: t.device,
    group_idx: int = 11,
    grouping: Tuple[Tuple[int, ...], ...] = ((0, 2),),
    use_grouping: bool = True,
    process: str = 'wjets'
) -> _component_collection:
    if process == 'wjets':
        return get_ff_dataset_with_qcd_weights_os(
            dataset=dataset,
            model=model,
            qcd_mask_os_loaded=qcd_mask_os_loaded,
            device=device,
            group_idx=group_idx,
            grouping=grouping,
            use_grouping=use_grouping,
            qcd_weight_binning=QCD_WEIGHT_BINNING,
            qcd_weight_n_bins=QCD_WEIGHT_N_BINS,
            qcd_weight_dynamic_delta=QCD_WEIGHT_DYNAMIC_DELTA,
            qcd_weight_dynamic_delta_last=QCD_WEIGHT_DYNAMIC_DELTA_LAST,
            qcd_weight_dynamic_min_qcd_yield=QCD_WEIGHT_DYNAMIC_MIN_QCD_YIELD,
        )
    elif process == "qcd":
        return get_ff_dataset_with_qcd_weights_ss(
            dataset=dataset,
            model=model,
            qcd_process_mask_ss_loaded=qcd_mask_os_loaded,
            device=device,
            njets_idx=group_idx,
            njets_groups=grouping,
            subtract_njets_based=use_grouping,
            qcd_weight_binning=QCD_WEIGHT_BINNING,
            qcd_weight_n_bins=QCD_WEIGHT_N_BINS,
            qcd_weight_dynamic_delta=QCD_SS_WEIGHT_DYNAMIC_DELTA,
            qcd_weight_dynamic_delta_last=QCD_SS_WEIGHT_DYNAMIC_DELTA_LAST,
            qcd_weight_dynamic_min_qcd_yield=QCD_SS_WEIGHT_DYNAMIC_MIN_QCD_YIELD,
        )


def find_dynamic_bin_edges(
    values_A: Union[np.ndarray, t.Tensor],
    weights_A: Union[np.ndarray, t.Tensor],
    values_B: Union[np.ndarray, t.Tensor],
    weights_B: Union[np.ndarray, t.Tensor],
    delta: float = 0.0,
    delta_last: float = 0.0,
    min_A_yield: float = 0.0,
    max_val: float = 1.0,
    min_val: float = 0.0,
) -> Union[np.ndarray, t.Tensor]:
    """
    Finds bin edges dynamically by walking from max_val down to min_val.
    Ensures that for each bin: Sum(w_A) >= min_A_yield AND Sum(w_A) - Sum(w_B) > delta.
    Compatible with both NumPy arrays and PyTorch tensors.
    """

    is_torch = isinstance(values_A, t.Tensor)

    if is_torch:
        values_all = t.cat([values_A, values_B])
        weights_net = t.cat([weights_A, -weights_B])
        weights_A_only = t.cat([weights_A, t.zeros_like(weights_B)])

        sort_idx = t.argsort(values_all, descending=True)

    else:
        values_all = np.concatenate([values_A, values_B])
        weights_net = np.concatenate([weights_A, -weights_B])
        weights_A_only = np.concatenate([weights_A, np.zeros_like(weights_B)])

        sort_idx = np.argsort(values_all)[::-1]

    weights_sorted = values_all[sort_idx]
    weights_net_sorted = weights_net[sort_idx]
    weights_A_only_sorted = weights_A_only[sort_idx]

    values_sorted_list = weights_sorted.tolist()
    weights_net_sorted_list = weights_net_sorted.tolist()
    weights_A_only_sorted_list = weights_A_only_sorted.tolist()

    # Top-Down Cumulative Sum Walking
    edges, accumulative_net, accumulative_A = [max_val], 0.0, 0.0
    for i in range(len(values_sorted_list)):
        if values_sorted_list[i] < min_val:
            break

        accumulative_net += weights_net_sorted_list[i]
        accumulative_A += weights_A_only_sorted_list[i]

        if accumulative_A >= min_A_yield and accumulative_net > delta:
            edges.append(values_sorted_list[i])
            accumulative_net, accumulative_A = 0.0, 0.0

    edges.append(min_val)

    # Upward merging if needed
    while len(edges) > 2:
        low, high = edges[-1], edges[-2]

        if is_torch:
            mask = (values_all >= low) & (values_all < high)
            final_bin_net = t.sum(weights_net[mask]).item()
        else:
            mask = (values_all >= low) & (values_all < high)
            final_bin_net = np.sum(weights_net[mask])

        if final_bin_net > delta_last:
            break

        edges.pop(-2)

    edges.reverse()

    if is_torch:
        return t.tensor(edges, dtype=values_A.dtype, device=values_A.device)
    else:
        return np.array(edges, dtype=values_A.dtype)

def build_qcd_weight_bins(
    qcd_values: t.Tensor,
    qcd_weights: t.Tensor,
    non_qcd_values: t.Tensor,
    non_qcd_weights: t.Tensor,
    binning: Literal['quantile', 'dynamic'] = 'quantile',
    n_bins: int = 10,
    dynamic_delta: float = 100.0,
    dynamic_delta_last: float = 100.0,
    dynamic_min_qcd_yield: float = 100.0,
) -> t.Tensor:
    if binning == 'quantile':
        bins = t.quantile(qcd_values, t.linspace(0, 1, n_bins + 1, device=qcd_values.device))
    elif binning == 'dynamic':
        min_val = t.minimum(qcd_values.min(), non_qcd_values.min()).item()
        max_val = t.maximum(qcd_values.max(), non_qcd_values.max()).item()
        bins = find_dynamic_bin_edges(
            values_A=qcd_values,
            weights_A=qcd_weights,
            values_B=non_qcd_values,
            weights_B=non_qcd_weights,
            delta=dynamic_delta,
            delta_last=dynamic_delta_last,
            min_A_yield=dynamic_min_qcd_yield,
            min_val=min_val,
            max_val=max_val,
        )
    else:
        raise ValueError(f"Unknown qcd binning option: {binning}")

    bins = t.unique(bins, sorted=True)
    if bins.numel() < 2:
        logger.warning("QCD bin builder returned <2 unique edges. Falling back to quantile binning.")
        bins = t.quantile(qcd_values, t.linspace(0, 1, n_bins + 1, device=qcd_values.device))
        bins = t.unique(bins, sorted=True)

    if bins.numel() < 2:
        raise RuntimeError("Could not construct valid QCD bin edges.")

    return bins


def get_ff_dataset_with_qcd_weights_os(
    dataset: _component_collection,
    model: t.nn.Module,
    qcd_mask_os_loaded: t.Tensor,
    device,
    group_idx: int = -1,
    grouping: Tuple[Tuple[int, ...], ...] = ((0, 2),),
    use_grouping: bool = True,
    qcd_weight_binning: Literal['quantile', 'dynamic'] = 'quantile',
    qcd_weight_n_bins: int = 10,
    qcd_weight_dynamic_delta: float = 100.0,
    qcd_weight_dynamic_delta_last: float = 100.0,
    qcd_weight_dynamic_min_qcd_yield: float = 100.0,
) -> _component_collection:
    """
    Build a dataset with OS QCD weights computed from SS control region shapes.

    Changes vs. previous version:
    - Replaces the mt_low_mask split by using dataset.SR_like.ss / dataset.SR_like.os.
    - For each njets group and SR_like value (True/False), compute QCD reweighting
      from SS (QCD-enriched) and apply the weights to OS in the same SR_like slice.
    """

    _dataset = deepcopy(dataset)

    # Optional: basic validation to ensure SR_like is precalculatesent
    if not hasattr(_dataset, "SR_like") or not hasattr(_dataset.SR_like, "ss") or not hasattr(_dataset.SR_like, "os"):
        raise AttributeError("Expected dataset.SR_like with .ss and .os boolean tensors.")

    # Initialize container for OS QCD weights
    _dataset.qcd_weights_os = t.full_like(
        _dataset.weights.os,
        fill_value=t.nan,
    )

    # --- predictions ---
    model.eval()
    with t.no_grad():
        prediction = deepcopy(_dataset.X)
        prediction.ss = predict_probabilities(model, _dataset.X.ss, device)
        prediction.os = predict_probabilities(model, _dataset.X.os, device)

    # --- original QCD masks ---
    qcd_process_mask_ss = _dataset.Y.ss == 2          # QCD in SS
    qcd_process_mask_os = qcd_mask_os_loaded          # QCD-like OS events (provided)

    def _group_mask(values: t.Tensor, group: Tuple[int, ...]) -> t.Tensor:
        if len(group) == 1:
            return values == group[0]
        return (values >= group[0]) & (values <= group[1])

    grouping_iter = grouping if use_grouping else ((0, 1000),)
    for current_group in grouping_iter:
        group_mask_ss = _group_mask(_dataset.X.ss[:, group_idx], current_group)
        group_mask_os = _group_mask(_dataset.X.os[:, group_idx], current_group)

        qcd_mask_ss = group_mask_ss & qcd_process_mask_ss
        non_qcd_mask_ss = group_mask_ss & ~qcd_process_mask_ss
        qcd_mask_os = group_mask_os & qcd_process_mask_os

        # --- split by SR_like True/False (replacing previous mt_low_mask split) ---
        for sr_value in (True, False):
            sr_mask_ss = (_dataset.SR_like.ss == sr_value)
            sr_mask_os = (_dataset.SR_like.os == sr_value)

            qcd_mask_ss_sr = qcd_mask_ss & sr_mask_ss
            non_qcd_mask_ss_sr = non_qcd_mask_ss & sr_mask_ss
            qcd_mask_os_sr = qcd_mask_os & sr_mask_os

            # skip empty regions
            if (
                qcd_mask_ss_sr.sum() == 0
                or non_qcd_mask_ss_sr.sum() == 0
                or qcd_mask_os_sr.sum() == 0
            ):
                continue

            bins = build_qcd_weight_bins(
                qcd_values=prediction.ss[qcd_mask_ss_sr].squeeze(),
                qcd_weights=_dataset.weights.ss[qcd_mask_ss_sr].squeeze(),
                non_qcd_values=prediction.ss[non_qcd_mask_ss_sr].squeeze(),
                non_qcd_weights=_dataset.weights.ss[non_qcd_mask_ss_sr].squeeze(),
                binning=qcd_weight_binning,
                n_bins=qcd_weight_n_bins,
                dynamic_delta=qcd_weight_dynamic_delta,
                dynamic_delta_last=qcd_weight_dynamic_delta_last,
                dynamic_min_qcd_yield=qcd_weight_dynamic_min_qcd_yield,
            )

            non_qcd_ss_hist, bins = t.histogram(
                input=prediction.ss[non_qcd_mask_ss_sr],
                bins=bins,
                weight=_dataset.weights.ss[non_qcd_mask_ss_sr],
            )

            # Build per-event weights for the OS slice using SS-derived bin subtraction.
            qcd_weights = _calculate_scaled_event_weights_generalized(
                prediction.os[qcd_mask_os_sr].squeeze(),
                t.ones_like(prediction.os[qcd_mask_os_sr].squeeze()),
                bins,
                non_qcd_ss_hist,
            )

            qcd_weights = set_negatives_to_one(qcd_weights)

            _dataset.qcd_weights_os[qcd_mask_os_sr] = qcd_weights
            _dataset.weights.os[qcd_mask_os_sr] = qcd_weights
            _dataset.class_weights.os[qcd_mask_os_sr] *= qcd_weights

    # --- relabel QCD events to background ---
    _dataset.class_weights.os = _dataset.weights.os
    _dataset.Y.os[qcd_mask_os_loaded] = 0

    _dataset.class_weights.os = _dataset.weights.os

    return _dataset.apply_func(
        lambda x: x.contiguous() if isinstance(x, t.Tensor) else x
    )



def get_ff_dataset_with_qcd_weights_ss(
    dataset: _component_collection,
    model: t.nn.Module,
    qcd_process_mask_ss_loaded: torch.Tensor,
    device,
    njets_idx: int = -1,
    njets_groups: Tuple[Tuple[int, ...], ...] = ((0,), (1,), (2, 100)),
    subtract_njets_based: bool = False,
    qcd_weight_binning: Literal['quantile', 'dynamic'] = 'quantile',
    qcd_weight_n_bins: int = 20,
    qcd_weight_dynamic_delta: float = 100.0,
    qcd_weight_dynamic_delta_last: float = 100.0,
    qcd_weight_dynamic_min_qcd_yield: float = 100.0,

) -> _component_collection:
    """
    Build a dataset where QCD weights are computed *only in the SS region*
    and saved to dataset.weights.ss.

    Differences from OS version:
    - Only SS quantities are used.
    - SR_like.ss determines the shape slices.
    - QCD weights are saved into weights.ss (and class_weights.ss if present).
    - No OS manipulation, no label rewriting.
    """

    _dataset = deepcopy(dataset)

    # Basic validation
    if not hasattr(_dataset, "SR_like") or not hasattr(_dataset.SR_like, "ss"):
        raise AttributeError("Expected dataset.SR_like.ss boolean tensor.")

    # Container for QCD weights in SS
    _dataset.qcd_weights_ss = torch.full_like(
        _dataset.weights.ss,
        fill_value=torch.nan,
    )

    # --- predictions ---
    model.eval()
    with torch.no_grad():
        prediction_ss = predict_probabilities(model, _dataset.X.ss, device)

    # --- masks ---
    qcd_mask_ss = qcd_process_mask_ss_loaded       # provided QCD-like SS mask
    non_qcd_mask_ss = ~qcd_mask_ss                 # everything else in SS

    # Loop over njets groups (or inclusive if not subtracting)
    for njets_group in (njets_groups if subtract_njets_based else ((0, 1000),)):
        # Define njets mask
        if len(njets_group) == 1:
            njets_mask_ss = _dataset.X.ss[:, njets_idx] == njets_group[0]
        else:
            njets_mask_ss = (
                (_dataset.X.ss[:, njets_idx] >= njets_group[0]) &
                (_dataset.X.ss[:, njets_idx] <= njets_group[1])
            )

        qcd_mask = qcd_mask_ss & njets_mask_ss
        non_qcd_mask = non_qcd_mask_ss & njets_mask_ss

        # --- split by SR_like.ss ---
        for sr_value in (True, False):
            sr_mask = (_dataset.SR_like.ss == sr_value)

            qcd_mask_sr = qcd_mask & sr_mask
            non_qcd_mask_sr = non_qcd_mask & sr_mask

            # Skip invalid regions
            if (
                qcd_mask_sr.sum() == 0
                or non_qcd_mask_sr.sum() == 0
            ):
                continue

            bins = build_qcd_weight_bins(
                qcd_values=prediction_ss[qcd_mask_sr].squeeze(),
                qcd_weights=_dataset.weights.ss[qcd_mask_sr].squeeze(),
                non_qcd_values=prediction_ss[non_qcd_mask_sr].squeeze(),
                non_qcd_weights=_dataset.weights.ss[non_qcd_mask_sr].squeeze(),
                binning=qcd_weight_binning,
                n_bins=qcd_weight_n_bins,
                dynamic_delta=qcd_weight_dynamic_delta,
                dynamic_delta_last=qcd_weight_dynamic_delta_last,
                dynamic_min_qcd_yield=qcd_weight_dynamic_min_qcd_yield,
            )

            logger.info(
                "QCD weight bins (%s, group=%s, SR_like=%s): %d",
                qcd_weight_binning,
                njets_group,
                sr_value,
                max(int(bins.numel()) - 1, 0),
            )

            non_qcd_hist, bins = t.histogram(
                input=prediction_ss[non_qcd_mask_sr],
                bins=bins,
                weight=_dataset.weights.ss[non_qcd_mask_sr],
            )

            # Compute QCD weights
            qcd_weights = _calculate_scaled_event_weights_generalized(
                prediction_ss[qcd_mask_sr].squeeze(),
                t.ones_like(prediction_ss[qcd_mask_sr].squeeze()),
                bins,
                non_qcd_hist,
            )

            qcd_weights = t.where(
                qcd_weights < 0,
                t.zeros_like(qcd_weights),
                qcd_weights,
            )

            # Save weights
            _dataset.weights.ss[qcd_mask_sr] = qcd_weights
            _dataset.qcd_weights_ss[qcd_mask_sr] = qcd_weights
            if hasattr(_dataset, "class_weights"):
                _dataset.class_weights.ss[qcd_mask_sr] *= qcd_weights

    missing_qcd_weights = qcd_mask_ss & torch.isnan(_dataset.qcd_weights_ss)
    if missing_qcd_weights.any():
        logger.warning(
            "Filling %d SS QCD weights with neutral scale factor 1.0 "
            "because no valid subtraction bin was available.",
            int(missing_qcd_weights.sum().item()),
        )
        neutral_weights = torch.ones_like(_dataset.qcd_weights_ss[missing_qcd_weights])
        _dataset.qcd_weights_ss[missing_qcd_weights] = neutral_weights
        _dataset.weights.ss[missing_qcd_weights] = neutral_weights

    return _dataset.apply_func(
        lambda x: x.contiguous() if isinstance(x, torch.Tensor) else x
    )



def _get_backend_and_device(tensor_or_array: Union[np.ndarray, t.Tensor]) -> tuple[Any, Any]:
    if isinstance(tensor_or_array, t.Tensor):
        return t, tensor_or_array.device
    elif isinstance(tensor_or_array, np.ndarray):
        return np, None
    else:
        raise TypeError(f"Input must be a NumPy array or PyTorch tensor, got {type(tensor_or_array)}")

def _calculate_scaled_event_weights_generalized(
    event_values: Union[np.ndarray, t.Tensor],
    event_original_weights: Union[np.ndarray, t.Tensor],
    bins: np.ndarray,
    total_subtraction_per_bin: Union[np.ndarray, t.Tensor],
) -> Union[np.ndarray, t.Tensor]:
    lib, device = _get_backend_and_device(event_values)
    is_torch = (lib == t)
    device_kwargs = {'device': device} if is_torch else {}

    raw = _collection(event_values, event_original_weights, total_subtraction_per_bin)
    
    initial = _collection(
        values=lib.asarray(raw.values, **device_kwargs),
        weights=lib.asarray(raw.weights, **device_kwargs),
        histograms=lib.asarray(raw.histograms, **device_kwargs)
    )
    
    shape_prefix = _collection(
        values=initial.values.shape[:-1],
        weights=initial.weights.shape[:-1],
        histograms=initial.histograms.shape[:-1]
    )

    bins = lib.asarray(bins, dtype=event_values.dtype, **device_kwargs)
    n_bins, n_events = len(bins) - 1, initial.values.shape[-1]

    flat = _collection(
        initial.values.reshape(-1, n_events).contiguous() if is_torch else initial.values.reshape(-1, n_events),
        initial.weights.reshape(-1, n_events),
        initial.histograms.reshape(-1, n_bins)
    )
    batch_size = _collection(
        values=flat.values.shape[0],
        weights=flat.weights.shape[0],
        histograms=flat.histograms.shape[0]
    )

    try:
        common_prefix_dim = np.broadcast_shapes(*shape_prefix.unrolled)
        max_batch_size = int(np.prod(common_prefix_dim)) if common_prefix_dim else 1
    except ValueError as e:
        raise ValueError(f"Prefix shapes {shape_prefix.unrolled} are not broadcastable. Error: {e}")

    if batch_size.values == 1 and max_batch_size > 1:
        flat.values = lib.broadcast_to(flat.values, (max_batch_size, n_events))
    if batch_size.weights == 1 and max_batch_size > 1:
        flat.weights = lib.broadcast_to(flat.weights, (max_batch_size, n_events))
    if batch_size.histograms == 1 and max_batch_size > 1:
        flat.histograms = lib.broadcast_to(flat.histograms, (max_batch_size, n_bins))

    _digitize, digitize_kwargs = (lib.bucketize, {'right': False}) if is_torch else (lib.digitize, {})
    raw_indices = _digitize(flat.values, bins, **digitize_kwargs) - 1

    # np.digitize places values equal to the final edge above the last bin,
    # whereas np.histogram includes that edge in the final bin.
    is_out_of_bounds = (flat.values < bins[0]) | (flat.values > bins[-1])
    event_bin_indices = lib.clip(raw_indices, 0, n_bins - 1)

    event_weights_for_summation = flat.weights.clone() if is_torch else flat.weights.copy()
    event_weights_for_summation[is_out_of_bounds] = 0.0  # Zero out weights for out-of-bounds events for sum calculation

    sum_original_weights_per_bin = lib.zeros((max_batch_size, n_bins), dtype=flat.weights.dtype, **device_kwargs)
    if is_torch:
        sum_original_weights_per_bin.scatter_add_(1, event_bin_indices.long(), event_weights_for_summation)
    else:
        for i in range(max_batch_size):
            sum_original_weights_per_bin[i] = lib.bincount(event_bin_indices[i], event_weights_for_summation[i], n_bins)

    scale_factor_per_bin = lib.ones_like(sum_original_weights_per_bin)
    non_zero_sum_mask = sum_original_weights_per_bin != 0

    scale_factor_per_bin[non_zero_sum_mask] = 1.0 - flat.histograms[non_zero_sum_mask] / sum_original_weights_per_bin[non_zero_sum_mask]

    zero_sum_non_zero_subtraction_mask = (sum_original_weights_per_bin == 0) & (flat.histograms != 0)
    scale_factor_per_bin[zero_sum_non_zero_subtraction_mask] = 0.0  # lib.nan

    # Gather Scale Factors for each Event
    if is_torch:
        scale_factors_for_events = lib.gather(scale_factor_per_bin, dim=1, index=event_bin_indices.long())
    else:
        row_idx_gather = lib.arange(max_batch_size)[:, None]
        scale_factors_for_events = scale_factor_per_bin[row_idx_gather, event_bin_indices]

    corrected_event_weights_flat = flat.weights * scale_factors_for_events
    corrected_event_weights_flat[is_out_of_bounds] = flat.weights[is_out_of_bounds]

    return corrected_event_weights_flat.reshape(initial.weights.shape)  # reshape back to original shape



@torch.no_grad()
def evaluate_binary_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    w_signal: float = 1.0,
    w_bkg: float = 1.0,
    w_qcd: float = 3.0,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    weight_sum = 0.0
    correct = 0
    total = 0

    for Xb, yb, wb, pb in loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        wb = wb.to(device, non_blocking=True)
        pb = pb.to(device, non_blocking=True)

        logits = model(Xb)
        y = yb.float().view(-1, 1)
        w = wb.float().view(-1, 1)

        is_signal = (y == 1).float()
        is_qcd = (pb == 2).float().view(-1, 1)
        is_bkg = (y == 0).float() * (1 - is_qcd)

        base_loss = criterion(logits.float(), y)
        loss_per_event = (
            w_signal * base_loss * is_signal
            + w_bkg * base_loss * is_bkg
            + w_qcd * base_loss * is_qcd
        )

        loss_sum += (loss_per_event * w).sum().item()
        weight_sum += w.sum().item()

        preds = (logits >= 0.5).float()
        correct += (preds.view(-1) == yb.view(-1)).sum().item()
        total += yb.numel()

    avg_loss = loss_sum / max(weight_sum, 1e-12)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy
