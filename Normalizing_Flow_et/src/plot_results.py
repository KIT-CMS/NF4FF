from __future__ import annotations
import math
import logging
import hashlib
import re
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
from NF_training_wjets import MLP, AffineCoupling, RealNVP
from NF_training_wjets import _component_collection
import correctionlib as cr
from classes.NeuralNetworks import RealNVP
from classes.Collection import (
    load_flow,
    load_model_config,
    evaluate_pdf,
    compute_eventwise_fake_factors,
)
from classes.Dataclasses import _component_collection
from classes.Plotting import (
    CMS_CHANNEL_TITLE,
    CMS_LABEL,
    CMS_LUMI_TITLE,
    CMS_NJETS_TITLE,
    reorder_for_rowwise_legend,
    adjust_ylim_for_legend,
)
from CustomLogging import setup_logging

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

    clip_value =  1/global_ff

    correction_factor = np.sum(ratio < clip_value)/len(ratio)
    global_ff_cor = global_ff / correction_factor

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
        fontsize=15,
        verticalalignment='top',
        fontproperties="Tex Gyre Heros:italic",
        bbox=dict(facecolor="white", alpha=0, edgecolor="white", boxstyle="round,pad=0.5"),
        transform=ax[0].transAxes
    )


def reorder_for_rowwise_legend(handles, labels, ncol, reverse=False):
    if reverse:
        handles = handles[::-1]
        labels = labels[::-1]

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

def equi_populated_bins(data, n_bins):

    data = np.asarray(data)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(data, quantiles)
    return bin_edges


def build_training_variables_tag(variables: list[str]) -> str:
    variables_joined = "|".join(variables)
    variables_hash = hashlib.sha1(variables_joined.encode("utf-8")).hexdigest()[:8]
    tail_variables = variables[4:]
    if tail_variables:
        readable_tail = "_".join(tail_variables)
        readable_tail = re.sub(r"[^A-Za-z0-9_]+", "_", readable_tail).strip("_")
    else:
        readable_tail = "none"
    return f"vars{len(variables)}_{readable_tail}_{variables_hash}"

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

def total_ff_corrected(df):
    df = df.copy()
    ff = cr.CorrectionSet.from_file('/work/mmoser/TauFakeFactors/workdir/ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz')

    frac = ff['process_fractions']


    ff_wjets = ff['Wjets_fake_factors']
    ff_qcd = ff['QCD_fake_factors']
    ff_ttbar = ff['ttbar_fake_factors']

    corr = cr.CorrectionSet.from_file('/work/mmoser/TauFakeFactors/workdir/ff_2026_01_19_check_variable/2018/FF_corrections_et.json.gz')


    df["wjets_classic_ff"] = ff_wjets.evaluate(
        df.pt_2.values,
        df.njets.values,
        df.pt_1.values,
        "nominal",
    )


    df['qcd_classic_ff'] = ff_qcd.evaluate(
        df.pt_2.values,
        df.njets.values,
        "nominal",
    )

    df['ttbar_classic_ff'] = ff_ttbar.evaluate(
        df.pt_2.values,
        df.njets.values,
        "nominal",
    )

    df["wjets_corrected_classic_ff"] = df["wjets_classic_ff"] * corr.compound["Wjets_compound_correction"].evaluate(
        df.tau_decaymode_2,
        df.eta_2,
        df.met,
        df.deltaR_ditaupair,
        df.pt_ttjj,
        df.mass_2,
        df.mt_tot,
        df.iso_1,
        df.m_vis,
        df.njets,
        "nominal",
    ) * corr["Wjets_DR_SR_correction"].evaluate(
        df.pt_tt,
        df.njets,
        "nominal",
    )

    df["qcd_corrected_classic_ff"] = df["qcd_classic_ff"] * corr.compound["QCD_compound_correction"].evaluate(
        df.tau_decaymode_2,
        df.eta_2,
        df.met,
        df.deltaR_ditaupair,
        df.pt_ttjj,
        df.mass_2,
        df.mt_tot,
        df.iso_1,
        df.m_vis,
        df.njets,
        "nominal",
    ) * corr["QCD_DR_SR_correction"].evaluate(
        df.pt_tt,
        df.njets,
        "nominal",
    )

    df["ttbar_corrected_classic_ff"] = df["ttbar_classic_ff"] * corr.compound["ttbar_compound_correction"].evaluate(
        df.tau_decaymode_2,
        df.eta_2,
        df.met,
        df.deltaR_ditaupair,
        df.pt_ttjj,
        df.mass_2,
        df.mt_tot,
        df.iso_1,
        df.m_vis,
        df.njets,
        "nominal",
    )

    df['process_fraction_wjets'] = frac.evaluate(
        'Wjets',
        df.mt_1.values,
        df.njets.values,
        'nominal'
    )

    df['process_fraction_qcd'] = frac.evaluate(
        'QCD',
        df.mt_1.values,
        df.njets.values,
        'nominal'
    )

    df['process_fraction_ttbar'] = frac.evaluate(
        'ttbar',
        df.mt_1.values,
        df.njets.values,
        'nominal'
    )

    df['corrected_ff'] = df['process_fraction_wjets'] * df['wjets_corrected_classic_ff'] + df['process_fraction_qcd'] * df['qcd_corrected_classic_ff'] + df['process_fraction_ttbar'] * df['ttbar_corrected_classic_ff']

    return df.copy()

def normalizing_flow_ff(df, variables, model_AR_like_wjets, model_SR_like_wjets, global_ff_wjets, model_AR_like_qcd, model_SR_like_qcd, global_ff_qcd, device,
                        plotting = False, plot_dir="plots"):
    df = df.copy()

    df_pt = get_my_data(df, variables).to_torch().to(device)

    pdf_AR_like_wjets = evaluate_pdf(model_AR_like_wjets, df_pt.X)
    pdf_SR_like_wjets = evaluate_pdf(model_SR_like_wjets, df_pt.X)


    ff_eventwise_full_wjets, ff_eventwise_clipped_wjets, global_ff_corr_wjets, clip_mask_wjets  = compute_eventwise_fake_factors(
        pdf_AR_like_wjets,
        pdf_SR_like_wjets,
        global_ff_wjets,
    )


    df = df[clip_mask_wjets]

    df_pt = get_my_data(df, variables).to_torch().to(device)

    pdf_AR_like_qcd = evaluate_pdf(model_AR_like_qcd, df_pt.X)
    pdf_SR_like_qcd = evaluate_pdf(model_SR_like_qcd, df_pt.X)

    ff_eventwise_full_qcd, ff_eventwise_clipped_qcd, global_ff_cor_qcd, clip_mask_qcd = compute_eventwise_fake_factors(
        pdf_AR_like_qcd,
        pdf_SR_like_qcd,
        global_ff_qcd,
    )

    ff_eventwise_clipped_wjets = ff_eventwise_clipped_wjets[clip_mask_qcd]
    df = df[clip_mask_qcd]
    if plotting == True:
        bins = np.logspace(-4,0, 21)
        fig, ax = plt.subplots(2, 1, figsize = (8,6))
        ax[0].set_title('Wjets eventwise FF')
        ax[0].hist(ff_eventwise_clipped_wjets, bins = bins, label = 'NF FF')
        ax[0].set_xscale('log')
        ax[1].set_title('QCD eventwise FF')
        ax[1].hist(ff_eventwise_clipped_qcd, bins = bins)
        ax[1].set_xscale('log')


    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))

    bins = np.logspace(np.log10(1e-3), np.log10(10), 50)

    plt.hist(ff_eventwise_full_qcd, bins=bins, color="royalblue", alpha=0.7)
    plt.hist(ff_eventwise_clipped_qcd, bins=bins, color="red", alpha=0.4)
    plt.xscale("log")
    plt.xlabel("Eventwise FF QCD")
    plt.ylabel("Counts")
    plt.title("Histogram on Logarithmic X-axis")

    plt.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(plot_dir / "hist_eventwise_ff_qcd.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))

    bins = np.logspace(np.log10(1e-3), np.log10(100), 50)

    plt.hist(ff_eventwise_full_wjets, bins=bins, color="royalblue", alpha=0.7)
    plt.hist(ff_eventwise_clipped_wjets, bins=bins, color="red", alpha=0.4)

    plt.xscale("log")
    plt.xlabel("Eventwise FF Wjets")
    plt.ylabel("Counts")
    plt.title("Histogram on Logarithmic X-axis")

    plt.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(plot_dir / "hist_eventwise_ff_wjets.png", dpi=200)
    plt.close()


    ff = cr.CorrectionSet.from_file('/work/mmoser/TauFakeFactors/workdir/ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz')

    frac = ff['process_fractions']


    ff_wjets = ff['Wjets_fake_factors']
    ff_qcd = ff['QCD_fake_factors']
    ff_ttbar = ff['ttbar_fake_factors']

    corr = cr.CorrectionSet.from_file('/work/mmoser/TauFakeFactors/workdir/ff_2026_01_19_check_variable/2018/FF_corrections_et.json.gz')


    df["wjets_classic_ff"] = ff_wjets.evaluate(
        df.pt_2.values,
        df.njets.values,
        df.pt_1.values,
        "nominal",
    )


    df['qcd_classic_ff'] = ff_qcd.evaluate(
        df.pt_2.values,
        df.njets.values,
        "nominal",
    )

    df['ttbar_classic_ff'] = ff_ttbar.evaluate(
        df.pt_2.values,
        df.njets.values,
        "nominal",
    )

    df["wjets_corrected_classic_ff"] = df["wjets_classic_ff"] * corr.compound["Wjets_compound_correction"].evaluate(
        df.tau_decaymode_2,
        df.eta_2,
        df.met,
        df.deltaR_ditaupair,
        df.pt_ttjj,
        df.mass_2,
        df.mt_tot,
        df.iso_1,
        df.m_vis,
        df.njets,
        "nominal",
    ) * corr["Wjets_DR_SR_correction"].evaluate(
        df.pt_tt,
        df.njets,
        "nominal",
    )

    df["qcd_corrected_classic_ff"] = df["qcd_classic_ff"] * corr.compound["QCD_compound_correction"].evaluate(
        df.tau_decaymode_2,
        df.eta_2,
        df.met,
        df.deltaR_ditaupair,
        df.pt_ttjj,
        df.mass_2,
        df.mt_tot,
        df.iso_1,
        df.m_vis,
        df.njets,
        "nominal",
    ) * corr["QCD_DR_SR_correction"].evaluate(
        df.pt_tt,
        df.njets,
        "nominal",
    )

    df["ttbar_corrected_classic_ff"] = df["ttbar_classic_ff"] * corr.compound["ttbar_compound_correction"].evaluate(
        df.tau_decaymode_2,
        df.eta_2,
        df.met,
        df.deltaR_ditaupair,
        df.pt_ttjj,
        df.mass_2,
        df.mt_tot,
        df.iso_1,
        df.m_vis,
        df.njets,
        "nominal",
    )

    df['process_fraction_wjets'] = frac.evaluate(
        'Wjets',
        df.mt_1.values,
        df.njets.values,
        'nominal'
    )

    df['process_fraction_qcd'] = frac.evaluate(
        'QCD',
        df.mt_1.values,
        df.njets.values,
        'nominal'
    )

    df['process_fraction_ttbar'] = frac.evaluate(
        'ttbar',
        df.mt_1.values,
        df.njets.values,
        'nominal'
    )
    df['ff_nf_wjets'] = df['process_fraction_wjets'] * ff_eventwise_clipped_wjets + df['process_fraction_qcd'] * ff_eventwise_clipped_qcd + df['process_fraction_ttbar'] * df['ttbar_corrected_classic_ff']

    return df

# ------- masks ----------

def mask_DR_wjets(df):                  # without SS/OS conditions !!!!!!!!!!!!11

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

def AR_like_qcd(df):

    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5) & (df.id_tau_vsJet_Tight_2 < 0.5))
    mask_a2 = (df.q_1 * df.q_2 > 0)
    mask_a4 = ((df.iso_1 > 0.02) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 < 50)
    mask_a7 = (df.process == 0)
    mask_DR = (mask_a1 & mask_a2 & mask_a4 & mask_a5 & mask_a6 & mask_a7)

    return df[mask_DR].copy()

def SR_like_qcd(df):

    mask_a1 = (df.id_tau_vsJet_Tight_2 > 0.5)
    mask_a2 = (df.q_1 * df.q_2 > 0)
    mask_a4 = ((df.iso_1 > 0.02) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 < 50)
    mask_a7 = (df.process == 0)
    mask_DR = (mask_a1 & mask_a2 & mask_a4 & mask_a5 & mask_a6 & mask_a7)

    return df[mask_DR].copy()


def mask_preselection_tight(df):
    mask_eta = (df.eta_1 <= 2.1) & (df.eta_2 <= 2.3)
    mask_pt = (df.pt_1 >= 40) & (df.pt_2 >= 35)
    #mask_m_vis = (df.m_vis >= 35)
    mask_tau_decay_mode = (df.tau_decaymode_2 == 0) | (df.tau_decaymode_2 == 1) | (df.tau_decaymode_2 == 10) | (df.tau_decaymode_2 == 11)
    return df[mask_eta & mask_pt &  mask_tau_decay_mode]

def SR(df):                 # without SS/OS conditions !!!!!!!!!!!!11
    mask_s1 = (df.id_tau_vsJet_Tight_2 > 0.5)
    mask_s2 = (df.nbtag == 0)
    mask_s3 = (df.q_1 * df.q_2 < 0)
    mask_s4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_s5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_s6 = (df.mt_1 < 70)

    mask_SR_like = (mask_s1 & mask_s2  & mask_s4 & mask_s5 & mask_s6)

    return(df[mask_SR_like])

def AR(df):                 # without SS/OS conditions !!!!!!!!!!!!11
    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5) & (df.id_tau_vsJet_Tight_2 < 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 < 70)

    mask_AR_like = (mask_a1 & mask_a2 & mask_a4 & mask_a5 & mask_a6)

    return(df[mask_AR_like])

def SR_like_wjets(df):
    mask_s1 = (df.id_tau_vsJet_Tight_2 > 0.5)
    mask_s2 = (df.nbtag == 0)
    mask_s4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_s5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_s6 = (df.mt_1 > 70)
    mask_s7 = (df.process == 0)

    mask_SR_like = (mask_s1 & mask_s2 & mask_s4 & mask_s5 & mask_s6 & mask_s7)

    return(df[mask_SR_like])

def AR_like_wjets(df):
    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5) & (df.id_tau_vsJet_Tight_2 < 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 > 70)
    mask_a7 = (df.process == 0)
    mask_AR_like = (mask_a1 & mask_a2 & mask_a4 & mask_a5 & mask_a6 & mask_a7)

    return(df[mask_AR_like])

# --------- loading variables ----------

with open('../configs/training_variables.yaml', 'r') as f:
    variables = yaml.safe_load(f)['variables']

dim = len(variables)
mode_dir = 'no_njets_split'

training_variables_tag = build_training_variables_tag(variables)
plot_root_dir = Path('plots') / mode_dir / f"training_{training_variables_tag}"
plot_root_dir.mkdir(parents=True, exist_ok=True)
category_plot_dir = plot_root_dir / f'inclusive_{mode_dir}_training'
category_plot_dir.mkdir(parents=True, exist_ok=True)

# Keep structure aligned with plot_results_njets.py (root + diagnostic/category folders)
(plot_root_dir / 'training_diagnostics').mkdir(parents=True, exist_ok=True)



# --------------- device -------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------load model -------------------

config_path = "../configs/config_NF.yaml"

config = load_model_config(config_path)

model_root_dir = Path('Training_results_new') / mode_dir / f"training_{training_variables_tag}"

chk_pth_model_AR_like_wjets = model_root_dir / 'Wjets' / 'all' / 'AR-like' / 'latest' / 'model_checkpoint.pth'
chk_pth_model_SR_like_wjets = model_root_dir / 'Wjets' / 'all' / 'SR-like' / 'latest' / 'model_checkpoint.pth'

chk_pth_model_AR_like_qcd = model_root_dir / 'QCD' / 'all' / 'AR-like' / 'latest' / 'model_checkpoint.pth'
chk_pth_model_SR_like_qcd = model_root_dir / 'QCD' / 'all' / 'SR-like' / 'latest' / 'model_checkpoint.pth'

model_AR_like_wjets = load_flow(dim=dim, cfg=config, checkpoint_path=str(chk_pth_model_AR_like_wjets), device=device)
model_SR_like_wjets = load_flow(dim=dim, cfg=config, checkpoint_path=str(chk_pth_model_SR_like_wjets), device=device)

model_AR_like_qcd = load_flow(dim=dim, cfg=config, checkpoint_path=str(chk_pth_model_AR_like_qcd), device=device)
model_SR_like_qcd = load_flow(dim=dim, cfg=config, checkpoint_path=str(chk_pth_model_SR_like_qcd), device=device)

# ----- load data -----

data_complete = pd.read_feather('../../data/data_complete.feather')

data_complete = data_complete[data_complete.njets == 2]

data_AR = AR(mask_preselection_tight(data_complete))                    # data + MC, without split in OS/SS           
data_AR = data_AR[data_AR.OS == True]
data_SR = SR(mask_preselection_tight(data_complete))

data_AR_OS = data_AR[(data_AR.process == 0)].copy()     # only data
data_AR_pt = get_my_data(data_AR_OS, variables).to_torch().to(device)   # torch dataset AR

data_AR_like_wjets = AR_like_wjets(mask_preselection_tight(data_complete))
data_SR_like_wjets = SR_like_wjets(mask_preselection_tight(data_complete))
data_AR_like_qcd = AR_like_qcd(mask_preselection_tight(data_complete))
data_SR_like_qcd = SR_like_qcd(mask_preselection_tight(data_complete))


data_AR_like_OS_wjets = data_AR_like_wjets[(data_AR_like_wjets.process == 0) & (data_AR_like_wjets.OS == True)]
data_SR_like_OS_wjets = data_SR_like_wjets[(data_SR_like_wjets.process == 0) & (data_SR_like_wjets.OS == True)]

data_AR_like_SS_qcd = data_AR_like_qcd[(data_AR_like_qcd.process == 0) & (data_AR_like_qcd.SS == True)]
data_SR_like_SS_qcd = data_SR_like_qcd[(data_SR_like_qcd.process == 0) & (data_SR_like_qcd.SS == True)]

data_SR_SS = data_SR[(data_SR.SS == True)]
data_SR_OS = data_SR[(data_SR.OS == True)]

# ------------ FF calculation ------------------

# -------- NF FF ------------

global_ff_wjets = len(data_SR_like_OS_wjets) / len(data_AR_like_OS_wjets)

global_ff_qcd = len(data_SR_like_SS_qcd) / len(data_AR_like_SS_qcd)



data_AR_OS_nf = normalizing_flow_ff(data_AR_OS, variables, model_AR_like_wjets, model_SR_like_wjets, global_ff_wjets, model_AR_like_qcd, model_SR_like_qcd, global_ff_qcd, device, plotting = True, plot_dir=category_plot_dir)

data_diboson_AR_OS = data_AR[((data_AR.process == 2) | (data_AR.process == 3)) ]
data_DY_AR_OS = data_AR[(data_AR.process == 4) | (data_AR.process == 5)]
data_ST_AR_OS = data_AR[(data_AR.process == 6) | (data_AR.process == 7)]
data_ttbar_L_AR_OS = data_AR[(data_AR.process == 9)]
data_embedding_AR_OS = data_AR[(data_AR.process == 10)]


data_diboson_AR_OS_nf = normalizing_flow_ff(data_diboson_AR_OS, variables, model_AR_like_wjets, model_SR_like_wjets, global_ff_wjets, model_AR_like_qcd, model_SR_like_qcd, global_ff_qcd, device, plot_dir=category_plot_dir)
data_DY_AR_OS_nf = normalizing_flow_ff(data_DY_AR_OS, variables, model_AR_like_wjets, model_SR_like_wjets, global_ff_wjets, model_AR_like_qcd, model_SR_like_qcd, global_ff_qcd, device, plot_dir=category_plot_dir)
data_ST_AR_OS_nf = normalizing_flow_ff(data_ST_AR_OS, variables, model_AR_like_wjets, model_SR_like_wjets, global_ff_wjets, model_AR_like_qcd, model_SR_like_qcd, global_ff_qcd, device, plot_dir=category_plot_dir)
data_ttbar_L_AR_OS_nf = normalizing_flow_ff(data_ttbar_L_AR_OS, variables, model_AR_like_wjets, model_SR_like_wjets, global_ff_wjets, model_AR_like_qcd, model_SR_like_qcd, global_ff_qcd, device, plot_dir=category_plot_dir)
data_embedding_AR_OS_nf = normalizing_flow_ff(data_embedding_AR_OS, variables, model_AR_like_wjets, model_SR_like_wjets, global_ff_wjets, model_AR_like_qcd, model_SR_like_qcd, global_ff_qcd, device, plot_dir=category_plot_dir)

data_events = data_SR_OS[(data_SR_OS.process == 0)]

data_diboson_SR_OS = data_SR_OS[(data_SR_OS.process == 2) | (data_SR_OS.process == 3)]
data_DY_SR_OS = data_SR_OS[(data_SR_OS.process == 4) | (data_SR_OS.process == 5)]
data_ST_SR_OS = data_SR_OS[(data_SR_OS.process == 6) | (data_SR_OS.process == 7)]
data_ttbar_L_SR_OS = data_SR_OS[(data_SR_OS.process == 9)]
data_embedding_SR_OS = data_SR_OS[(data_SR_OS.process == 10)]

# --------------- corr classic FF ------------------

data_AR_OS_classic = total_ff_corrected(data_AR_OS)
data_diboson_AR_OS_classic = total_ff_corrected(data_diboson_AR_OS)
data_DY_AR_OS_classic = total_ff_corrected(data_DY_AR_OS)
data_ST_AR_OS_classic = total_ff_corrected(data_ST_AR_OS)
data_ttbar_L_AR_OS_classic = total_ff_corrected(data_ttbar_L_AR_OS)
data_embedding_AR_OS_classic = total_ff_corrected(data_embedding_AR_OS)

# ------------ plotting -------

list_variables = ['pt_1', 'pt_2', 'm_vis', 'deltaR_ditaupair', 'pt_vis', 'pt_tt', 'tau_decaymode_2','m_fastmtt', 
                  'eta_1', 'eta_2', 'jeta_1', 'jeta_2', 'deltaR_12j1', 'deltaR_12j2', 'deltaR_12jj', 'deltaR_1j1', 'deltaR_1j2',
                  'deltaR_2j1', 'deltaR_2j2', 'deltaR_jj', 'met', 'mt_1', 'mt_2', 'mt_tot', 'iso_1', 'iso_2']

with open('../configs/labels.yaml', 'r') as f:
    labels = yaml.safe_load(f)
labels = labels['et']
list_xlabels = [labels[k] for k in list_variables]
#list_xlabels = [r"Electron $\mathrm{p}_{T}$ / GeV", r"Hadronic Tau $\mathrm{p}_{T}$ / GeV", r"Visible di-$\mathrm{\tau}$ mass / GeV", r"$\Delta R(e, ^{}\tau_{h})$", r"Visible di-$\tau \mathrm{p}_{T}$ / GeV"]

list_bins = [
    np.linspace(0, 150, 31),
    np.linspace(0, 150, 31),
    np.linspace(0, 220, 31),
    np.linspace(0, 5, 21),
    np.linspace(0, 160, 31),
    np.linspace(0, 160, 31),
    np.linspace(0, 12, 13),
    np.linspace(0, 220, 31),
    np.linspace(-3, 3, 31),
    np.linspace(-3, 3, 31),
    np.linspace(-3, 3, 31),
    np.linspace(-3, 3, 31),
    np.linspace(0, 5, 21),
    np.linspace(0, 5, 21),
    np.linspace(0, 5, 21),
    np.linspace(0, 5, 21),
    np.linspace(0, 5, 21),
    np.linspace(0, 5, 21),
    np.linspace(0, 5, 21),
    np.linspace(0, 5, 21),
    np.linspace(0, 220, 31),            #met
    np.linspace(0, 70, 31),            #mt_1
    np.linspace(0, 70, 31),            #mt_2
    np.linspace(0, 70, 31),            #mt_tot
    np.linspace(0, 0.2, 31),
    np.linspace(0.6, 1.2, 31),
]


for var, bins, xlabel in zip(list_variables, list_bins, list_xlabels):



    counts_ff_data_classic, bin_edges = np.histogram(data_AR_OS_classic[var], weights=data_AR_OS_classic['corrected_ff'], bins = bins)
    counts_ff_data_classic2, bin_edges = np.histogram(data_AR_OS_classic[var], weights=data_AR_OS_classic['corrected_ff']**2, bins = bins)

    counts_ff_diboson_classic, _ = np.histogram(data_diboson_AR_OS_classic[var], weights = data_diboson_AR_OS_classic.weight * data_diboson_AR_OS_classic['corrected_ff'], bins = bins)
    counts_ff_diboson_classic2, _ = np.histogram(data_diboson_AR_OS_classic[var], weights = (data_diboson_AR_OS_classic.weight * data_diboson_AR_OS_classic['corrected_ff'])**2, bins = bins)
    counts_ff_DY_classic, _ = np.histogram(data_DY_AR_OS_classic[var], weights = data_DY_AR_OS_classic.weight * data_DY_AR_OS_classic['corrected_ff'], bins = bins)
    counts_ff_DY_classic2, _ = np.histogram(data_DY_AR_OS_classic[var], weights = (data_DY_AR_OS_classic.weight * data_DY_AR_OS_classic['corrected_ff'])**2, bins = bins)
    counts_ff_ST_classic, _ = np.histogram(data_ST_AR_OS_classic[var], weights = data_ST_AR_OS_classic.weight * data_ST_AR_OS_classic['corrected_ff'], bins = bins)
    counts_ff_ST_classic2, _ = np.histogram(data_ST_AR_OS_classic[var], weights = (data_ST_AR_OS_classic.weight * data_ST_AR_OS_classic['corrected_ff'])**2, bins = bins)
    counts_ff_ttbar_L_classic, _ = np.histogram(data_ttbar_L_AR_OS_classic[var], weights = data_ttbar_L_AR_OS_classic.weight * data_ttbar_L_AR_OS_classic['corrected_ff'], bins = bins)
    counts_ff_ttbar_L_classic2, _ = np.histogram(data_ttbar_L_AR_OS_classic[var], weights = (data_ttbar_L_AR_OS_classic.weight * data_ttbar_L_AR_OS_classic['corrected_ff'])**2, bins = bins)
    counts_ff_embedding_classic, _ = np.histogram(data_embedding_AR_OS_classic[var], weights = data_embedding_AR_OS_classic.weight * data_embedding_AR_OS_classic['corrected_ff'], bins = bins)
    counts_ff_embedding_classic2, _ = np.histogram(data_embedding_AR_OS_classic[var], weights = (data_embedding_AR_OS_classic.weight * data_embedding_AR_OS_classic['corrected_ff'])**2, bins = bins)

    counts_FF_classic = counts_ff_data_classic - counts_ff_diboson_classic - counts_ff_DY_classic - counts_ff_ST_classic - counts_ff_ttbar_L_classic - counts_ff_embedding_classic

    counts_ff_data, bin_edges = np.histogram(data_AR_OS_nf[var], weights=data_AR_OS_nf['ff_nf_wjets'], bins = bins)
    counts_ff_data2, _ = np.histogram(data_AR_OS_nf[var], weights=data_AR_OS_nf['ff_nf_wjets']**2, bins = bins)

    counts_ff_diboson, _ = np.histogram(data_diboson_AR_OS_nf[var], weights = data_diboson_AR_OS_nf.weight * data_diboson_AR_OS_nf['ff_nf_wjets'], bins = bins)
    counts_ff_diboson2, _ = np.histogram(data_diboson_AR_OS_nf[var], weights = (data_diboson_AR_OS_nf.weight* data_diboson_AR_OS_nf['ff_nf_wjets'])**2, bins = bins)
    counts_ff_DY, _ = np.histogram(data_DY_AR_OS_nf[var], weights = data_DY_AR_OS_nf.weight * data_DY_AR_OS_nf['ff_nf_wjets'], bins = bins)
    counts_ff_DY2 , _ =  np.histogram(data_DY_AR_OS_nf[var], weights = (data_DY_AR_OS_nf.weight * data_DY_AR_OS_nf['ff_nf_wjets'])**2, bins = bins)
    counts_ff_ST, _ = np.histogram(data_ST_AR_OS_nf[var], weights = data_ST_AR_OS_nf.weight * data_ST_AR_OS_nf['ff_nf_wjets'], bins = bins)
    counts_ff_ST2, _ = np.histogram(data_ST_AR_OS_nf[var], weights = (data_ST_AR_OS_nf.weight * data_ST_AR_OS_nf['ff_nf_wjets'])**2, bins = bins)
    counts_ff_ttbar_L, _ = np.histogram(data_ttbar_L_AR_OS_nf[var], weights = data_ttbar_L_AR_OS_nf.weight * data_ttbar_L_AR_OS_nf['ff_nf_wjets'], bins = bins)
    counts_ff_ttbar_L2, _ = np.histogram(data_ttbar_L_AR_OS_nf[var], weights = (data_ttbar_L_AR_OS_nf.weight * data_ttbar_L_AR_OS_nf['ff_nf_wjets'])**2, bins = bins)
    counts_ff_embedding, _ = np.histogram(data_embedding_AR_OS_nf[var], weights = data_embedding_AR_OS_nf.weight * data_embedding_AR_OS_nf['ff_nf_wjets'], bins = bins)
    counts_ff_embedding2, _ = np.histogram(data_embedding_AR_OS_nf[var], weights = (data_embedding_AR_OS_nf.weight * data_embedding_AR_OS_nf['ff_nf_wjets'])**2, bins = bins)
    counts_FF = counts_ff_data - counts_ff_diboson - counts_ff_DY - counts_ff_ST - counts_ff_ttbar_L - counts_ff_embedding
    print(counts_FF)
    # ---------------- MC part ---------------

    counts_diboson, _ = np.histogram(data_diboson_SR_OS[var], weights = data_diboson_SR_OS.weight, bins = bins)
    counts_diboson2, _ = np.histogram(data_diboson_SR_OS[var], weights = data_diboson_SR_OS.weight**2, bins = bins)
    counts_DY, _ = np.histogram(data_DY_SR_OS[var], weights = data_DY_SR_OS.weight, bins = bins)
    counts_DY2, _ = np.histogram(data_DY_SR_OS[var], weights = data_DY_SR_OS.weight**2, bins = bins)
    counts_ST, _ = np.histogram(data_ST_SR_OS[var], weights = data_ST_SR_OS.weight, bins = bins)
    counts_ST2, _ = np.histogram(data_ST_SR_OS[var], weights = data_ST_SR_OS.weight**2, bins = bins)
    counts_ttbar_L, _ = np.histogram(data_ttbar_L_SR_OS[var], weights = data_ttbar_L_SR_OS.weight, bins = bins)
    counts_ttbar_L2, _ = np.histogram(data_ttbar_L_SR_OS[var], weights = data_ttbar_L_SR_OS.weight**2, bins = bins)
    counts_embedding, _ = np.histogram(data_embedding_SR_OS[var], weights = data_embedding_SR_OS.weight, bins = bins)
    counts_embedding2, _ = np.histogram(data_embedding_SR_OS[var], weights = data_embedding_SR_OS.weight**2, bins = bins)

    # --------------- data part -----------

    counts_data, _ = np.histogram(data_events[var], bins = bins)


    bin_widths = np.diff(bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    fig, ax = plt.subplots(
        4, 1,
        figsize=(9,9),
        sharex=True,
        gridspec_kw={'height_ratios': [4,1,0.2,1], 'hspace': 0.05},
        constrained_layout=True

    )

    CMS_CHANNEL_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_LABEL(ax)
    #CMS_NJETS_TITLE(ax)

    # X and Y error
    y_error = np.sqrt(counts_data)
    x_error = 0.5*bin_widths
    num = np.sqrt(
    counts_ff_data2 + counts_ff_diboson2 + counts_ff_ttbar_L2 +
    counts_ff_embedding2 + counts_ff_ST2 + counts_ff_DY2 +
    counts_diboson2 + counts_ttbar_L2 + counts_embedding2 +
    counts_DY2 + counts_ST2
    )

    den = (
    counts_FF + counts_diboson + counts_ttbar_L +
    counts_embedding + counts_ST + counts_DY
    )

    y_error_stat = np.divide(num, den, out=np.zeros_like(num), where=den!=0)


    num_classic = np.sqrt(
        counts_ff_data_classic2 + counts_ff_diboson_classic2 +
        counts_ff_ttbar_L_classic2 + counts_ff_embedding_classic2 +
        counts_ff_ST_classic2 + counts_ff_DY_classic2 +
        counts_diboson2 + counts_ttbar_L2 + counts_embedding2 +
        counts_DY2 + counts_ST2
    )

    den_classic = (
        counts_FF_classic + counts_diboson + counts_ttbar_L +
        counts_embedding + counts_ST + counts_DY
    )

    y_error_stat_classic = np.divide(
        num_classic,
        den_classic,
        out=np.zeros_like(num_classic),
        where=den_classic!=0
    )


    # --- Upper panel: stacked histograms + data ---
    ax[0].bar(bin_centers, counts_diboson, width = bin_widths, color = "#94a4a2", label = 'Diboson')
    ax[0].bar(bin_centers, counts_ttbar_L, bottom = counts_diboson, width = bin_widths,color = '#832db6', label = r'$t\bar{t} \to \tau$')
    ax[0].bar(bin_centers, counts_ST, bottom = counts_diboson + counts_ttbar_L, width = bin_widths,color = "#717581", label = r"Single t")
    ax[0].bar(bin_centers, counts_DY, bottom = counts_diboson + counts_ttbar_L + counts_ST, width = bin_widths,color = '#3f90da', label = r'$Z \to \ell \ell$')
    ax[0].bar(bin_centers, counts_FF, bottom = counts_diboson + counts_ttbar_L + counts_ST + counts_DY,width = bin_widths, color = "#a96b59", label = r'Jet $\rightarrow \tau_h$')
    ax[0].bar(bin_centers, counts_embedding, bottom = counts_diboson + counts_ttbar_L + counts_ST + counts_DY + counts_FF,width = bin_widths, 
            color = '#ffa90e', label = r'$\tau$ embedded')

    ax[0].errorbar(bin_centers, counts_data, yerr=y_error, xerr=x_error,fmt='o', color='black', label='Data', markersize=5)
    ax[0].set_ylabel("Events")
    handles, labels = ax[0].get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    handles, labels = reorder_for_rowwise_legend(handles, labels, ncol = 4)
    ax[0].legend(
        handles,
        labels,
        title=' ',
        title_fontsize=20,
        loc='upper left',
        ncol=4,
        frameon=False
    )
    adjust_ylim_for_legend(ax[0])
    # Remove top ticks
    ax[0].tick_params(direction='in', top=True, right=True)

    # --- Lower panel: ratio plot ---
    ax[1].errorbar(bin_centers, np.divide(counts_data, den, out=np.zeros_like(counts_data, dtype = float), where=den != 0), 
                    xerr=x_error,
                    yerr = np.divide(y_error,counts_data, out = np.zeros_like(counts_data, dtype = float), where = counts_data != 0),
                    fmt='o', color='black', markersize=5,
                    label = r'NF $F_\text{F}$')
    ax[1].fill_between(
    bin_centers,
    1 - y_error_stat,
    1 + y_error_stat,
    color="gray",
    alpha=0.3,
    step='mid',
    label="Stat. Unc.")
    ax[1].axhline(1, color='red', linestyle='--', linewidth=1.5)
    ax[1].set_ylabel("Data / Model")
    ax[1].set_ylim([0.5, 1.5])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc= 'upper right',ncol = 2)

    ax[2].axis('off')
    # --- Lower panel: ratio plot ---
    ax[3].errorbar(bin_centers, np.divide(counts_data, den_classic, out=np.zeros_like(counts_data, dtype = float), where=den != 0), 
                    xerr=x_error,
                    yerr = np.divide(y_error,counts_data,out = np.zeros_like(counts_data, dtype = float), where= counts_data != 0),
                    fmt='o', color='black', markersize=5,
                    label = r'Cor class $F_\text{F}$ ')
    ax[3].fill_between(
    bin_centers,
    1 - y_error_stat_classic,
    1 + y_error_stat_classic,
    color="gray",
    alpha=0.3,
    step='mid',
    label="Stat. Unc.")
    ax[3].axhline(1, color='red', linestyle='--', linewidth=1.5)
    ax[3].set_ylabel("Data / Model")
    ax[3].set_ylim([0.5, 1.5])
    ax[3].grid(True, linestyle=':', alpha=0.7)
    ax[3].tick_params(direction='in', top=True, right=True)
    ax[3].legend(loc = 'lower right',ncol = 2)
    ax[3].set_xlabel(xlabel)
    #fig.subplots_adjust(bottom = 0.12)

    fig.savefig(category_plot_dir / f'{var}.png')
    fig.savefig(category_plot_dir / f'{var}.pdf')
