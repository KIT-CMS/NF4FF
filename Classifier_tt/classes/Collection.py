import numpy as np
import pandas as pd
import random
import torch as t
import torch.nn as nn
import yaml
import math
import matplotlib.pyplot as plt
from typing import Any, Literal, Union, Tuple
from sklearn.model_selection import train_test_split
from classes.models import BinaryClassifier
import classes.helper as helper



# ----- seeds -----

SEED = 42
t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ---------- data loading ----------

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(
    input_dim: int,
    checkpoint_path: str,
    device: t.device,
) -> BinaryClassifier:
    model = BinaryClassifier(input_dim).to(device)
    checkpoint = t.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model



# ---------- data processing ----------



@t.no_grad()
def predict_probabilities(
    model: nn.Module,
    X: t.Tensor,
    device: t.device,
) -> t.Tensor:
    X = X.to(device, non_blocking=True)
    logits = model(X)
    return logits.squeeze(1).cpu()

def set_negatives_to_one(tensor):
    # Using torch.where
    return t.where(tensor < 0, t.ones_like(tensor), tensor)

def equal_frequency_bins(x, n_bins):
    quantiles = np.linspace(0, 1, n_bins + 1)
    return np.quantile(x, quantiles)

def get_class_weights(
    weights: t.Tensor,
    Y: t.Tensor,
    classes: tuple = (0, 1),
    class_weighted: bool = True,
) -> t.Tensor:

    weights = weights.float()
    Y = Y.long()

    _weights = t.zeros_like(weights)

    total_weight = weights.sum()

    for _class in classes:
        mask = (Y == _class)
        class_sum = weights[mask].sum()

        if class_sum > 0:
            _weights[mask] = total_weight / class_sum
        else:
            _weights[mask] = 0.0

    return _weights * (weights if class_weighted else 1.0)

def equi_populated_bins(data, n_bins):

    data = np.asarray(data)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(data, quantiles)
    return bin_edges

def split_even_odd(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold1 = df[df.event_var  == 0].reset_index(drop=True)
    fold2 = df[df.event_var == 1].reset_index(drop=True)

    train1, val1 = train_test_split(
        fold1, test_size=0.5, random_state=SEED
    )
    train2, val2 = train_test_split(
        fold2, test_size=0.5, random_state=SEED
    )

    return train1.reset_index(drop=True), val1.reset_index(drop=True), train2.reset_index(drop=True), val2.reset_index(drop=True)

# ----- Data utilities -----


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

    raw = helper._collection(event_values, event_original_weights, total_subtraction_per_bin)
    
    initial = helper._collection(
        values=lib.asarray(raw.values, **device_kwargs),
        weights=lib.asarray(raw.weights, **device_kwargs),
        histograms=lib.asarray(raw.histograms, **device_kwargs)
    )
    
    shape_prefix = helper._collection(
        values=initial.values.shape[:-1],
        weights=initial.weights.shape[:-1],
        histograms=initial.histograms.shape[:-1]
    )

    bins = lib.asarray(bins, dtype=event_values.dtype, **device_kwargs)
    n_bins, n_events = len(bins) - 1, initial.values.shape[-1]

    flat = helper._collection(
        initial.values.reshape(-1, n_events).contiguous() if is_torch else initial.values.reshape(-1, n_events),
        initial.weights.reshape(-1, n_events),
        initial.histograms.reshape(-1, n_bins)
    )
    batch_size = helper._collection(
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

    is_out_of_bounds = (raw_indices < 0) | (raw_indices >= n_bins)
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

# ----- QCD weight binning -----



def refresh_qcd_weights(
    dataset: helper._component_collection,
    model: nn.Module,
    qcd_mask_os_loaded: t.Tensor,
    device: t.device,
) -> helper._component_collection:
    return get_ff_dataset_with_qcd_weights_os(
        dataset=dataset,
        model=model,
        qcd_mask_os_loaded=qcd_mask_os_loaded,
        device=device,
        njets_idx=11,
        njets_groups=((0,), (1,), (2, 100)),
        subtract_njets_based=True,
        reweight_njets_based=True,
        qcd_weight_binning=QCD_WEIGHT_BINNING,
        qcd_weight_n_bins=QCD_WEIGHT_N_BINS,
        qcd_weight_dynamic_delta=QCD_WEIGHT_DYNAMIC_DELTA,
        qcd_weight_dynamic_delta_last=QCD_WEIGHT_DYNAMIC_DELTA_LAST,
        qcd_weight_dynamic_min_qcd_yield=QCD_WEIGHT_DYNAMIC_MIN_QCD_YIELD,
    )



def get_ff_dataset_with_qcd_weights_os(
    dataset: helper._component_collection,
    model: t.nn.Module,
    qcd_mask_os_loaded: t.Tensor,
    device,
    njets_idx: int = -1,
    njets_groups: Tuple[Tuple[int, ...], ...] = ((0,), (1,), (2, 100)),
    subtract_njets_based: bool = False,
    reweight_njets_based: bool = True,
    qcd_weight_binning: Literal['quantile', 'dynamic'] = 'quantile',
    qcd_weight_n_bins: int = 10,
    qcd_weight_dynamic_delta: float = 100.0,
    qcd_weight_dynamic_delta_last: float = 100.0,
    qcd_weight_dynamic_min_qcd_yield: float = 100.0,
) -> helper._component_collection:
    """
    Build a dataset with OS QCD weights computed from SS control region shapes.

    Changes vs. previous version:
    - Replaces the mt_low_mask split by using dataset.SR_like.ss / dataset.SR_like.os.
    - For each njets group and SR_like value (True/False), compute QCD reweighting
      from SS (QCD-enriched) and apply the weights to OS in the same SR_like slice.
    """

    _dataset = deepcopy(dataset)

    # Optional: basic validation to ensure SR_like is present
    if not hasattr(_dataset, "SR_like") or not hasattr(_dataset.SR_like, "ss") or not hasattr(_dataset.SR_like, "os"):
        raise AttributeError("Expected dataset.SR_like with .ss and .os boolean tensors.")

    # Initialize container for OS QCD weights
    _dataset.qcd_weights_os = torch.full_like(
        _dataset.weights.os,
        fill_value=torch.nan,
    )

    # --- predictions ---
    model.eval()
    with torch.no_grad():
        prediction = deepcopy(_dataset.X)
        prediction.ss = predict_probabilities(model, _dataset.X.ss, device)
        prediction.os = predict_probabilities(model, _dataset.X.os, device)

    # --- original QCD masks ---
    qcd_process_mask_ss = _dataset.Y.ss == 2          # QCD in SS
    qcd_process_mask_os = qcd_mask_os_loaded          # QCD-like OS events (provided)

    # Loop over njets groups (or single inclusive bin if not subtract_njets_based)
    for njets_group in (njets_groups if subtract_njets_based else ((0, 1000),)):
        if len(njets_group) == 1:
            njets_mask_ss = _dataset.X.ss[:, njets_idx] == njets_group[0]
            njets_mask_os = _dataset.X.os[:, njets_idx] == njets_group[0]
        else:
            njets_mask_ss = (
                (_dataset.X.ss[:, njets_idx] >= njets_group[0]) &
                (_dataset.X.ss[:, njets_idx] <= njets_group[1])
            )
            njets_mask_os = (
                (_dataset.X.os[:, njets_idx] >= njets_group[0]) &
                (_dataset.X.os[:, njets_idx] <= njets_group[1])
            )

        qcd_mask_ss = njets_mask_ss & qcd_process_mask_ss
        non_qcd_mask_ss = njets_mask_ss & ~qcd_process_mask_ss
        qcd_mask_os = njets_mask_os & qcd_process_mask_os

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

            logger.info(
                "QCD weight bins (%s, njets=%s, SR_like=%s): %d",
                qcd_weight_binning,
                njets_group,
                sr_value,
                max(int(bins.numel()) - 1, 0),
            )

            non_qcd_ss_hist, bins = t.histogram(
                input=prediction.ss[non_qcd_mask_ss_sr],
                bins=bins,
                weight=_dataset.weights.ss[non_qcd_mask_ss_sr],
            )

            # Compute QCD scaling weights from SS and apply to OS in same SR_like slice
            qcd_weights = _calculate_scaled_event_weights_generalized(
                prediction.ss[qcd_mask_ss_sr].squeeze(),
                t.ones_like(prediction.ss[qcd_mask_ss_sr].squeeze()),
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

    # --- njets-based class weights (same logic as before) ---
    njets_classes = t.zeros_like(qcd_mask_os_loaded, dtype=torch.long)
    for idx, njets_group in enumerate(njets_groups if reweight_njets_based else ((0, 1000),)):
        if len(njets_group) == 1:
            njets_mask = _dataset.X.os[:, njets_idx] == njets_group[0]
        else:
            njets_mask = (
                (_dataset.X.os[:, njets_idx] >= njets_group[0]) &
                (_dataset.X.os[:, njets_idx] <= njets_group[1])
            )
        njets_classes[njets_mask] = idx

    _dataset.class_weights.os = _dataset.weights.os

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

    is_out_of_bounds = (raw_indices < 0) | (raw_indices >= n_bins)
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



# ------ plotting -----

def CMS_CHANNEL_TITLE(ax, *args, **kwargs):
    ax[0].set_title(
        r"$e\tau_h$",
        fontsize=20,
        loc="left",
        fontproperties="Tex Gyre Heros",
    )

def CMS_NJETS_TITLE(ax, *args, **kwargs):
    ax[0].set_title(
        r"$N_{jets}$ inclusive",
        fontsize=20,
        loc="center",
        fontproperties="Tex Gyre Heros"
    )

def CMS_LUMI_TITLE(ax, *args, **kwargs):
    ax[0].set_title(
        "59.8 $fb^{-1}$ (2018, 13 TeV)",
        fontsize=20,
        loc="right",
        fontproperties="Tex Gyre Heros"
    )

def CMS_LABEL(ax, *args, **kwargs):
    ax[0].text(
        0.025, 0.95,
        "Private work (CMS data/simulation)",
        fontsize=20,
        verticalalignment='top',
        fontproperties="Tex Gyre Heros:italic",
        bbox=dict(facecolor="white", alpha=0, edgecolor="white", boxstyle="round,pad=0.5"),
        transform=ax[0].transAxes
    )

def reorder_for_rowwise_legend(handles, labels, ncol):
    n = len(handles)
    nrows = math.ceil(n / ncol)

    new_handles, new_labels = [], []

    for col in range(ncol):
        for row in range(nrows):
            idx = row * ncol + col
            if idx < n:
                new_handles.append(handles[idx])
                new_labels.append(labels[idx])

    return new_handles, new_labels

def adjust_ylim_for_legend(ax=None, spacing=0.05):
    if ax is None:
        ax = plt.gca()

    fig = ax.figure
    fig.canvas.draw()

    if (leg := ax.get_legend()) is None:
        return

    bbox_leg, bbox_ax = leg.get_window_extent(), ax.get_window_extent()

    legend_height_ratio = bbox_leg.height / bbox_ax.height

    ymin, ymax = ax.get_ylim()
    scale = ax.get_yscale()

    if (available_fraction := 1.0 - legend_height_ratio - spacing) <= 0.1:
        available_fraction = 0.1

    if scale == "linear":
        data_max_y = ax.dataLim.y1
        data_range = data_max_y - ymin
        new_range = data_range / available_fraction
        new_ymax = ymin + new_range
        ax.set_ylim(ymin, new_ymax)

    elif scale == "log":
        log_ymin = np.log10(ymin)
        log_data_max = np.log10(ax.dataLim.y1)
        log_range = log_data_max - log_ymin
        new_log_range = log_range / available_fraction
        new_log_ymax = log_ymin + new_log_range

        new_log_ymax = np.ceil(new_log_ymax)

        new_ymax = 10 ** new_log_ymax
        ax.set_ylim(ymin, new_ymax)
