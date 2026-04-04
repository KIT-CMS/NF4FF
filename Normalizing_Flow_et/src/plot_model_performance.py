from __future__ import annotations
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from tap import Tap
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Union
from classes.config_loader import load_config
from classes.path_managment import StorePathHelper
from CustomLogging import setup_logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from classes.config_loader import load_config
from classes.training import train_epoch, val_epoch
from classes.path_managment import StorePathHelper
import logging
from CustomLogging import setup_logging
import yaml
import numpy as np
import random
from contextlib import contextmanager
from tap import Tap
from typing import Literal, Generator
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Any, Dict
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch as t
import CODE.HELPER as helper
import logging
import random
from dataclasses import KW_ONLY, dataclass
from typing import (Any, Callable, Dict, Iterable, List, Protocol, Tuple,
                    Union, runtime_checkable)
from contextlib import contextmanager
from typing import (Any, Callable, Dict, Generator, Iterable, Iterator, List,
                    Optional, Tuple, Type, Union, get_args, get_origin)
from sklearn.model_selection import train_test_split

from CustomLogging import setup_logging
import time
from NF_training_wjets import MLP, AffineCoupling, RealNVP
from NF_training_wjets import _component_collection
import correctionlib as cr
from matplotlib.ticker import ScalarFormatter
import numpy as np

import torch as t
from dataclasses import dataclass
from typing import Any, Union, Generator
from contextlib import contextmanager
import random

@dataclass
class ModelConfig:
    n_layers: int
    hidden_dims: int
    s_scale: float

@dataclass

class _same_sign_opposite_sign_split(metaclass=helper.CollectionMeta):
    ss: Union[t.Tensor, pd.DataFrame, np.ndarray]
    os: Union[t.Tensor, pd.DataFrame, np.ndarray]


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
class _collection:
    values: Any
    weights: Any
    histograms: Any
    
    @property
    def unrolled(self) -> tuple[Any, ...]:
        return (self.values, self.weights, self.histograms)


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


def CMS_CHANNEL_TITLE(ax, *args, **kwargs):
    ax[0].set_title(
        r"$e\tau_h$",
        fontsize=20,
        loc="left",
        fontproperties="Tex Gyre Heros"
    )

def CMS_NJETS_TITLE(ax, *args, **kwargs):
    ax[0].set_title(
        r"$N_{jets} \geq 0$",
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



def mask_DR_wjets(df):

    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 > 70)
    mask_DR = (mask_a1 & mask_a2  & mask_a4 & mask_a5 & mask_a6)

    return df[mask_DR].copy()

def mask_DR_qcd(df):

    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5))
    mask_a2 = (df.q_1 * df.q_2 > 0)
    mask_a4 = ((df.iso_1 > 0.02) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 < 50)
    mask_DR = (mask_a1 & mask_a2 & mask_a4 & mask_a5 & mask_a6)

    return df[mask_DR].copy()

def mask_preselection_loose(df):
    mask_eta = (df.eta_1 <= 2.1) & (df.eta_2 <= 2.3)
    mask_pt = (df.pt_1 >= 33) & (df.pt_2 >= 30)
    mask_tau_decay_mode = (df.tau_decaymode_2 == 0) | (df.tau_decaymode_2 == 1) | (df.tau_decaymode_2 == 10) | (df.tau_decaymode_2 == 11)
    return df[mask_eta & mask_pt & mask_tau_decay_mode]


def SR_like(df):
    mask_s1 = (df.id_tau_vsJet_Tight_2 > 0.5)
    return(df[mask_s1])

# AR-like mask
def AR_like(df):
    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5) & (df.id_tau_vsJet_Tight_2 < 0.5))

    return(df[mask_a1])


# SR-like masks
def SR_like_SS(df):
    mask_s1 = (df.id_tau_vsJet_Tight_2 > 0.5)
    mask_s2 = (df.nbtag == 0)
    mask_s3 = ((df.q_1 * df.q_2) > 0)
    mask_s4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_s5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_s6 = (df.mt_1 >= 70)

    mask_SR_like = (mask_s1 & mask_s2 & mask_s3 & mask_s4 & mask_s5 & mask_s6)

    return(df[mask_SR_like])

# AR-like mask
def AR_like_SS(df):
    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5) & (df.id_tau_vsJet_Tight_2 < 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a3 = ((df.q_1 * df.q_2) > 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 >= 70)

    mask_AR_like = (mask_a1 & mask_a2 & mask_a3 & mask_a4 & mask_a5 & mask_a6)

    return(df[mask_AR_like])


# SR masks
def SR(df):
    mask_s1 = (df.id_tau_vsJet_Tight_2 > 0.5)
    mask_s2 = (df.nbtag == 0)
    mask_s3 = ((df.q_1 * df.q_2) < 0)
    mask_s4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_s5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_s6 = (df.mt_1 < 70)

    mask_SR_like = (mask_s1 & mask_s2 & mask_s3 & mask_s4 & mask_s5 & mask_s6)

    return(df[mask_SR_like])

# AR mask
def AR(df):
    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5) & (df.id_tau_vsJet_Tight_2 < 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a3 = ((df.q_1 * df.q_2) < 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 < 70)

    mask_AR_like = (mask_a1 & mask_a2 & mask_a3 & mask_a4 & mask_a5 & mask_a6)

    return(df[mask_AR_like])

# SR masks
def SR_SS(df):
    mask_s1 = (df.id_tau_vsJet_Tight_2 > 0.5)
    mask_s2 = (df.nbtag == 0)
    mask_s3 = ((df.q_1 * df.q_2) > 0)
    mask_s4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_s5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_s6 = (df.mt_1 < 70)

    mask_SR_like = (mask_s1 & mask_s2 & mask_s3 & mask_s4 & mask_s5 & mask_s6)

    return(df[mask_SR_like])

# AR mask
def AR_SS(df):
    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5) & (df.id_tau_vsJet_Tight_2 < 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a3 = ((df.q_1 * df.q_2) > 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 < 70)

    mask_AR_like = (mask_a1 & mask_a2 & mask_a3 & mask_a4 & mask_a5 & mask_a6)

    return(df[mask_AR_like])


def get_my_data_qcd(df, training_var):
    _df = df

    return _component_collection(
        X=_df[training_var].to_numpy(dtype=np.float32),
        weights=_df["weight_qcd"].to_numpy(dtype=np.float32),

    )
def get_my_data_wjets(df, training_var):
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

with open('../configs/training_variables.yaml', 'r') as f:
    variables = yaml.safe_load(f)['variables']

dim = len(variables)




# --------------------------- data loading ----------------------

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data_complete = pd.read_feather('../../data/data_complete.feather')

data_DR_qcd = mask_DR_qcd(data_complete)

data_DR_qcd = mask_preselection_loose(data_DR_qcd)

data_DR_qcd_test2 = data_DR_qcd[data_DR_qcd.event_var == 1].reset_index(drop=True)

data_AR_like_qcd_test2 = AR_like(data_DR_qcd_test2)
data_SR_like_qcd_test2 = SR_like(data_DR_qcd_test2)

data_AR_like_qcd_test2 = data_AR_like_qcd_test2[(data_AR_like_qcd_test2.process == 0) & (data_AR_like_qcd_test2.SS == True)].reset_index(drop=True)
data_SR_like_qcd_test2 = data_SR_like_qcd_test2[(data_SR_like_qcd_test2.process == 0) & (data_SR_like_qcd_test2.SS == True)].reset_index(drop=True)

data_AR_like_pt_qcd_test2 = get_my_data_qcd(data_AR_like_qcd_test2, variables).to_torch().to(device)
data_SR_like_pt_qcd_test2 = get_my_data_qcd(data_SR_like_qcd_test2, variables).to_torch().to(device)

print((data_AR_like_pt_qcd_test2.weights))


data_DR_wjets = mask_DR_wjets(data_complete)

data_DR_wjets = mask_preselection_loose(data_DR_wjets)

data_DR_wjets_test2 = data_DR_wjets[data_DR_wjets.event_var == 1].reset_index(drop=True)

data_AR_like_wjets_test2 = AR_like(data_DR_wjets_test2)
data_SR_like_wjets_test2 = SR_like(data_DR_wjets_test2)

data_AR_like_wjets_test2 = data_AR_like_wjets_test2[(data_AR_like_wjets_test2.process == 0) & (data_AR_like_wjets_test2.OS == True)].reset_index(drop=True)
data_SR_like_wjets_test2 = data_SR_like_wjets_test2[(data_SR_like_wjets_test2.process == 0) & (data_SR_like_wjets_test2.OS == True)].reset_index(drop=True)

data_AR_like_pt_wjets_test2 = get_my_data_wjets(data_AR_like_wjets_test2, variables).to_torch().to(device)
data_SR_like_pt_wjets_test2 = get_my_data_wjets(data_SR_like_wjets_test2, variables).to_torch().to(device)

config_path = "../configs/config_NF.yaml"

config = load_model_config(config_path)


chk_pth_model_AR_like_wjets = 'Training_results_new/Wjets/all/AR-like/latest/model_checkpoint.pth'
chk_pth_model_SR_like_wjets = 'Training_results_new/Wjets/all/SR-like/latest/model_checkpoint.pth'

chk_pth_model_AR_like_qcd = 'Training_results_new/QCD/all/AR-like/latest/model_checkpoint.pth'
chk_pth_model_SR_like_qcd = 'Training_results_new/QCD/all/SR-like/latest/model_checkpoint.pth'


model_AR_like_qcd = load_flow(dim = dim, cfg = config, checkpoint_path = chk_pth_model_AR_like_qcd, device = device)
model_SR_like_qcd = load_flow(dim = dim, cfg = config, checkpoint_path = chk_pth_model_SR_like_qcd, device = device)

model_AR_like_wjets = load_flow(dim = dim, cfg = config, checkpoint_path = chk_pth_model_AR_like_wjets, device = device)
model_SR_like_wjets = load_flow(dim = dim, cfg = config, checkpoint_path = chk_pth_model_SR_like_wjets, device = device)



# ----- sampling and plotting -----

n_samples = 1000000
x_samples_AR_like_qcd = model_AR_like_qcd.sample(n_samples).detach().cpu().numpy()
x_samples_SR_like_qcd = model_SR_like_qcd.sample(n_samples).detach().cpu().numpy()
dim = data_SR_like_pt_qcd_test2.X.shape[1]

var_AR_like_qcd = [data_AR_like_pt_qcd_test2.X.detach().cpu().numpy().T[i] for i in range(dim)]
weights_AR_like_qcd = data_AR_like_pt_qcd_test2.weights.detach().cpu().numpy()

var_SR_like_qcd = [data_SR_like_pt_qcd_test2.X.detach().cpu().numpy().T[i] for i in range(dim)]
weights_SR_like_qcd = data_SR_like_pt_qcd_test2.weights.detach().cpu().numpy()

x_samples_AR_like_wjets = model_AR_like_wjets.sample(n_samples).detach().cpu().numpy()
x_samples_SR_like_wjets = model_SR_like_wjets.sample(n_samples).detach().cpu().numpy()

var_AR_like_wjets = [data_AR_like_pt_wjets_test2.X.detach().cpu().numpy().T[i] for i in range(dim)]
weights_AR_like_wjets = data_AR_like_pt_wjets_test2.weights.detach().cpu().numpy()

var_SR_like_wjets = [data_SR_like_pt_wjets_test2.X.detach().cpu().numpy().T[i] for i in range(dim)]
weights_SR_like_wjets = data_SR_like_pt_wjets_test2.weights.detach().cpu().numpy()




bins_pt_1 = np.linspace(20, 100, 50)
bins_pt_2 = np.linspace(20, 100, 50)
bins_m_vis = np.linspace(0, 300, 50)
bins_deltaR = np.linspace(0, 6, 50)
bins_mff = np.linspace(0,250,50)
bins_pt_tt = np.linspace(0, 160, 50)
bins_iso_1 = np.linspace(0,1,50)
bins_met = np.linspace(0, 250, 50)
bins_ptvis = np.linspace(0,250,50)
bins = [bins_pt_1, bins_pt_2, bins_m_vis, bins_deltaR, bins_met, bins_ptvis, bins_mff]

with open('../configs/labels.yaml', 'r') as f:
    labels = yaml.safe_load(f)
labels = labels['et']
variables_plotting = [labels[k] for k in variables]

for i in range(dim):

    fig, ax = plt.subplots(2,2, figsize = (10,8))

    fig.text(0.04, 0.75, "AR-like", va='center', rotation='vertical', fontsize=20)
    fig.text(0.04, 0.30, "SR-like", va='center', rotation='vertical', fontsize=20)
    ax[0, 0].set_title("QCD")
    ax[0][0].hist(x_samples_AR_like_qcd.T[i], bins = bins[i], alpha = 0.5, label = f'flow AR-like', density=True)
    ax[0][0].hist(var_AR_like_qcd[i], weights = weights_AR_like_qcd, bins = bins[i], alpha=0.5, label = f'data AR_like', density=True)
    ax[0][0].set_ylabel('density')
    ax[0][0].legend()
    adjust_ylim_for_legend(ax[0][0])

    ax[1][0].hist(x_samples_SR_like_qcd.T[i], bins = bins[i], alpha = 0.5, label = f'flow SR_like', density=True)
    ax[1][0].hist(var_SR_like_qcd[i], weights = weights_SR_like_qcd, bins = bins[i], alpha=0.5, label = f'data SR_like', density=True)
    ax[1][0].set_xlabel(variables_plotting[i])
    ax[1][0].set_ylabel('density')
    ax[1][0].legend()
    adjust_ylim_for_legend(ax[1][0])

    ax[0, 1].set_title("W+jets")
    ax[0][1].hist(x_samples_AR_like_wjets.T[i], bins = bins[i], alpha = 0.5, label = f'flow AR-like', density=True)
    ax[0][1].hist(var_AR_like_wjets[i], weights = weights_AR_like_wjets, bins = bins[i], alpha=0.5, label = f'data AR_like', density=True)
    ax[0][1].legend()
    adjust_ylim_for_legend(ax[0][1])

    ax[1][1].hist(x_samples_SR_like_wjets.T[i], bins = bins[i], alpha = 0.5, label = f'flow SR_like', density=True)
    ax[1][1].hist(var_SR_like_wjets[i], weights = weights_SR_like_wjets, bins = bins[i], alpha=0.5, label = f'data SR_like', density=True)
    ax[1][1].set_xlabel(variables_plotting[i])
    ax[1][1].legend()
    adjust_ylim_for_legend(ax[1][1])
    fig.savefig(f'plots_performance/var_{i}.png')
