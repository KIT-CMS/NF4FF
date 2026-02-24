from __future__ import annotations
import math
import logging
from dataclasses import dataclass, KW_ONLY
from pathlib import Path
from typing import (Literal, Tuple, Any, List, Union, Callable, Dict, 
                    Generator, Iterable, Iterator, Optional, Type, 
                    get_args, get_origin, Protocol, runtime_checkable)
from contextlib import contextmanager
import pandas as pd
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from tap import Tap
import yaml
import random
from sklearn.model_selection import train_test_split
from classes.config_loader import load_config
from classes.path_managment import StorePathHelper
from classes.training import train_epoch, val_epoch
from CustomLogging import setup_logging
import CODE.HELPER as helper
import time
from Normalizing_Flow.src.NF_training_wjets import MLP, AffineCoupling, RealNVP
from Normalizing_Flow.src.NF_training_wjets import _component_collection
import correctionlib as cr
from matplotlib.ticker import ScalarFormatter

matplotlib.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})


# -------- data classes


@contextmanager
def rng_seed(seed: int) -> Generator[None, None, None]:
    np_rng_state, py_rng_state = np.random.get_state(), random.getstate()
    t_rng_state = t.get_rng_state()

    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_rng_state)
        random.setstate(py_rng_state)
        t.set_rng_state(t_rng_state)


@dataclass
class ModelConfig:
    n_layers: int
    hidden_dims: int
    s_scale: float

@dataclass

class _same_sign_opposite_sign_split(metaclass=helper.CollectionMeta):
    ss: Union[t.Tensor, pd.DataFrame, np.ndarray]
    os: Union[t.Tensor, pd.DataFrame, np.ndarray]


@dataclass
class _collection:
    values: Any
    weights: Any
    histograms: Any
    
    @property
    def unrolled(self) -> tuple[Any, ...]:
        return (self.values, self.weights, self.histograms)

# ------------ functions ----------

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

def load_model_config(path: str) -> ModelConfig:
    cfg = load_config(path)
    return ModelConfig(
        n_layers=cfg["model"]["n_layers"],
        hidden_dims=cfg["model"]["hidden_dims"],
        s_scale=cfg["model"]["s_scale"],
    )

def load_flow(
    dim: int,
    cfg: ModelConfig,
    checkpoint_path: str,
    device: torch.device,
) -> RealNVP:

    model = RealNVP(
        dim=dim,
        n_layers=cfg.n_layers,
        hidden_dims=(cfg.hidden_dims,),
        s_scale=cfg.s_scale,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


@torch.no_grad()
def evaluate_pdf(model: RealNVP, X: torch.Tensor) -> np.ndarray:
    """Returns PDF evaluated at events"""
    # Use model(X) so the model can apply the scaler and add the scaling Jacobian
    log_pdf = model(X)
    print(f"log_pdf min: {log_pdf.min()}, max: {log_pdf.max()}, mean: {log_pdf.mean()}")
    log_pdf = torch.clamp(log_pdf, min=-1e10)
    pdf = torch.exp(log_pdf).cpu().numpy()
    print(f"pdf min: {pdf.min()}, max: {pdf.max()}, mean: {pdf.mean()}")
    return pdf



def compute_eventwise_fake_factors(
    pdf_AR: np.ndarray,
    pdf_SR: np.ndarray,
    global_ff: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:

    # ---- safe ratio
    ratio = np.divide(
        pdf_SR,
        np.maximum(pdf_AR, 1e-100),
        out=np.zeros_like(pdf_SR, dtype=float),
        where=(pdf_AR > 0) & (pdf_SR > 0)
)

    # ---- quantile clipping
    q = 0.99
    clip_value = np.quantile(ratio, q)
    global_ff_cor = global_ff / q

    # ---- full FF for all events
    ff_eventwise_full = global_ff_cor * ratio

    # ---- mask for clipping and nonzero PDFs
    clip_mask = (ratio <= clip_value) & (pdf_AR > 0) & (pdf_SR > 0)

    # ---- clipped FF
    ff_eventwise_clipped = ff_eventwise_full[clip_mask]

    return ff_eventwise_full, ff_eventwise_clipped, global_ff_cor, clip_mask


def get_my_data(df, training_var):
    _df = df

    return _component_collection(
        X=_df[training_var].to_numpy(dtype=np.float32),
        weights=_df["weight_wjets"].to_numpy(dtype=np.float32),

    )

@dataclass
class _same_sign_opposite_sign_split(metaclass=helper.CollectionMeta):
    ss: Union[torch.Tensor, pd.DataFrame, np.ndarray]
    os: Union[torch.Tensor, pd.DataFrame, np.ndarray]


@dataclass
class _component_collection(metaclass=helper.CollectionMeta):
    _: KW_ONLY
    X: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    weights: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None


variables = [
    'pt_1',
    'pt_2',
    'm_vis',
    'deltaR_ditaupair',
    'eta_1',
    'eta_2',
]

dim = len(variables)



# ----------------- masks ---------------

def mask_DR(df):

    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 > 70)
    mask_DR = (mask_a1 & mask_a2  & mask_a4 & mask_a5 & mask_a6)

    return df[mask_DR].copy()

def mask_preselection_tight(df):
    mask_eta = (df.eta_1 <= 2.1) & (df.eta_2 <= 2.3)
    mask_pt = (df.pt_1 >= 40) & (df.pt_2 >= 35)
    mask_tau_decay_mode = (df.tau_decaymode_2 == 0) | (df.tau_decaymode_2 == 1) | (df.tau_decaymode_2 == 10) | (df.tau_decaymode_2 == 11)
    return df[mask_eta & mask_pt & mask_tau_decay_mode]

def SR(df):
    mask_s1 = (df.id_tau_vsJet_Tight_2 > 0.5)
    mask_s2 = (df.nbtag == 0)
    mask_s4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_s5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_s6 = (df.mt_1 < 70)

    mask_SR_like = (mask_s1 & mask_s2 & mask_s4 & mask_s5 & mask_s6)

    return(df[mask_SR_like])

def AR(df):
    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5) & (df.id_tau_vsJet_Tight_2 < 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 < 70)

    mask_AR_like = (mask_a1 & mask_a2 & mask_a4 & mask_a5 & mask_a6)

    return(df[mask_AR_like])

def SR_like(df):
    mask_s1 = (df.id_tau_vsJet_Tight_2 > 0.5)
    mask_s2 = (df.nbtag == 0)
    mask_s4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_s5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_s6 = (df.mt_1 > 70)

    mask_SR_like = (mask_s1 & mask_s2 & mask_s4 & mask_s5 & mask_s6)

    return(df[mask_SR_like])

def AR_like(df):
    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5) & (df.id_tau_vsJet_Tight_2 < 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 > 70)

    mask_AR_like = (mask_a1 & mask_a2 & mask_a4 & mask_a5 & mask_a6)

    return(df[mask_AR_like])

def equi_populated_bins(data, n_bins):

    data = np.asarray(data)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(data, quantiles)
    return bin_edges

def _get_qcd_weights_from_SS(
        variable: str,
        data_SS: pd.DataFrame,
        nbins: int = 20,
):
    bins = equi_populated_bins(nbins)
    MC_counts, _ = np.histogram(data_SS[variable][data_SS.process > 0], weights = data_SS.weights[data_SS.process == 0], bins = bins)
    qcd_weights = _calculate_scaled_event_weights_generalized(
        event_values = data_SS[variable][data_SS.process == 0],
        event_original_weights= data_SS.weights[data_SS.process == 0],
        bins = bins,
        total_subtraction_per_bin=MC_counts
    )
    
    return data_SS[variable][data_SS.process == 0], qcd_weights




# ------------------------------

def main():

    # --------------- device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------load model

    config_path = "../configs/config_NF.yaml"

    config = load_model_config(config_path)

    chk_pth_model_AR_like = 'Training_results_new/all/AR-like/2026-02-16/0_14-27-21/model_checkpoint.pth'
    chk_pth_model_SR_like = 'Training_results_new/all/SR-like/2026-02-16/0_16-23-38/model_checkpoint.pth'

    model_AR_like = load_flow(dim = 6, cfg = config, checkpoint_path = chk_pth_model_AR_like, device = device)
    model_SR_like = load_flow(dim = 6, cfg = config, checkpoint_path = chk_pth_model_SR_like, device = device)

    # ----------- load data

    data_complete = pd.read_feather('../../data/data_complete.feather')

    # fold 2 test data

    data_complete = data_complete[data_complete.event_var == 1]

    data_AR = AR(mask_preselection_tight(data_complete)) # without split in OS/SS
    
    data_AR_OS = data_AR[(data_AR.process == 0) & (data_AR.OS == True)]

    data_AR_pt = get_my_data(data_AR_OS).to_torch().to(device)

    data_AR_like = AR_like(mask_preselection_tight(data_complete))
    data_SR_like = SR_like(mask_preselection_tight(data_complete))

    data_AR_like_OS = data_AR_like[(data_AR_like.process == 0) & (data_AR_like.OS == True)]
    data_SR_like_OS = data_SR_like[(data_SR_like.process == 0) & (data_SR_like.OS == True)]

    # -------------- calculate eventwise FFs ------------


    pdf_AR_like = evaluate_pdf(model_AR_like, data_AR_pt.X)
    pdf_SR_like = evaluate_pdf(model_SR_like, data_AR_pt.X)

    global_ff = len(data_SR_like_OS) / len(data_AR_like_OS)

    ff_eventwise_full, ff_eventwise_clipped, global_ff_corr, clip_mask  = compute_eventwise_fake_factors(
        pdf_AR_like,
        pdf_SR_like,
        global_ff,
    )

    data_AR_clipped = data_AR_OS[clip_mask]

    assert len(ff_eventwise_clipped) == len(data_AR_clipped)
