import logging
import math
import os
import random
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union

import CODE.HELPER as helper
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch as t
import torch.nn as nn
from CustomLogging import setup_logging
from sklearn.model_selection import train_test_split
from tap import Tap

from training_wjets import BinaryClassifier

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

logger = setup_logging(logger=logging.getLogger(__name__))

VARIABLES_WJETS = [
    "pt_1", "pt_2", "eta_1", "eta_2", "jpt_1", "jpt_2", "jeta_1", "jeta_2",
    "m_fastmtt", "pt_fastmtt", "met", "njets", "mt_tot", "m_vis",
    "pt_tt", "pt_vis", "mjj", "pt_dijet", "pt_ttjj", "deltaEta_jj", "deltaR_jj",
    "deltaR_ditaupair", "deltaR_1j1", "deltaR_1j2",
    "deltaR_2j1", "deltaR_2j2", "deltaR_12j1", "deltaR_12j2", "deltaEta_1j1",
    "deltaEta_1j2", "deltaEta_2j1", "deltaEta_2j2", "deltaEta_12j1", "deltaEta_12j2",
    "tau_decaymode_1", "tau_decaymode_2", "nbtag",
]
VARIABLES_QCD = VARIABLES_WJETS[:-1]

PROCESS_LABELS = {
    0: "QCD",
    1: "Wjets",
    2: "diboson_J",
    3: "diboson_L",
    4: "DYjets_J",
    5: "DYjets_L",
    6: "ST_J",
    7: "ST_L",
    8: "ttbar_J",
    9: "ttbar_L",
    10: "embedding",
}

PROCESS_COLORS = {
    0: "#b9ac70",
    1: "#e76300",
    2: "#9f887e",
    3: "#94a4a2",
    4: "#b9ac70",
    5: "#3f90da",
    6: "#717581",
    7: "#5882ae",
    8: "#964c88",
    9: "#615fc8",
    10: "#ffa90e",
}

CHANNEL_CONFIG = {
    "wjets": {
        "input_dim": len(VARIABLES_WJETS),
        "variables": VARIABLES_WJETS,
        "process_order": [1, 10, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        "compact_groups": [
            (r"W+jets", [1], "#e76300"),
            (r"$\tau$ embedded", [10], "#ffa90e"),
            (r"Diboson", [2, 3], "#b9ac70"),
            (r"Jet$\rightarrow \tau_{h}$", [4, 5], "#717581"),
            (r"Single t", [6, 7], "#717581"),
            (r"$t\bar{t}$", [8, 9], "#832db6"),
            (r"QCD multijet", [0], "#b9ac70"),
        ],
        "default_ckpt_fold1": "Classifier/results/Wjets/inclusive/fold1/last/",
        "default_ckpt_fold2": "Classifier/results/Wjets/inclusive/fold2/last/",
        "saved_weight_files": ("qcd_weights_train.pt", "qcd_weights_val.pt"),
        "saved_weight_column": "weight_wjets",
        "saved_weight_process": 0,
        "target_process": 1,
        "reduced_plot_label": "MC W+jets",
        "data_region": "os",
        "model_region": "os",
        "output_prefix": "wjets",
    },
    "qcd": {
        "input_dim": len(VARIABLES_QCD),
        "variables": VARIABLES_QCD,
        "process_order": [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9],
        "compact_groups": [
            (r"QCD multijet", [0], "#b9ac70"),
            (r"W+jets", [1], "#e76300"),
            (r"$\tau$ embedded", [10], "#ffa90e"),
            (r"Diboson", [2, 3], "#b9ac70"),
            (r"Jet$\rightarrow \tau_{h}$", [4, 5], "#717581"),
            (r"Single t", [6, 7], "#717581"),
            (r"$t\bar{t}$", [8, 9], "#832db6"),
        ],
        "default_ckpt_fold1": "Classifier/results/QCD/inclusive/fold1/last/",
        "default_ckpt_fold2": "Classifier/results/QCD/inclusive/fold2/last/",
        "saved_weight_files": ("qcd_weights_qcd_train.pt", "qcd_weights_qcd_val.pt"),
        "saved_weight_column": "weight_qcd",
        "saved_weight_process": 0,
        "target_process": 0,
        "reduced_plot_label": "QCD",
        "data_region": "ss",
        "model_region": "ss",
        "output_prefix": "qcd",
    },
}


class Args(Tap):
    channel: Literal["wjets", "qcd", "both"] = "both"
    bins: Literal["equi_populated", "uniform"] = "equi_populated"
    n_bins: int = 20
    data_complete_path: str = "data/data_complete.feather"
    output_dir: str = "plots"
    ckpt_pth_fold1: str = ""
    ckpt_pth_fold2: str = ""
    write_back: bool = False
    plot_process_fractions: bool = False


@dataclass
class _same_sign_opposite_sign_split(metaclass=helper.CollectionMeta):
    ss: Union[torch.Tensor, pd.DataFrame, np.ndarray]
    os: Union[torch.Tensor, pd.DataFrame, np.ndarray]


@dataclass
class _component_collection(metaclass=helper.CollectionMeta):
    _: KW_ONLY
    X: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    Y: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    Label: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    weights: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    class_weights: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    process: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None


@dataclass
class _collection:
    values: Any
    weights: Any
    histograms: Any

    @property
    def unrolled(self) -> tuple[Any, ...]:
        return (self.values, self.weights, self.histograms)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_model(input_dim: int, checkpoint_path: Path, device: torch.device) -> BinaryClassifier:
    model = BinaryClassifier(input_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_probabilities(model: nn.Module, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    if X.shape[0] == 0:
        return torch.empty((0,), dtype=torch.float32)
    X = X.to(device, non_blocking=True)
    logits = model(X)
    return logits.squeeze(1).cpu()


def _get_backend_and_device(tensor_or_array: Union[np.ndarray, t.Tensor]) -> tuple[Any, Any]:
    if isinstance(tensor_or_array, t.Tensor):
        return t, tensor_or_array.device
    if isinstance(tensor_or_array, np.ndarray):
        return np, None
    raise TypeError(f"Input must be a NumPy array or PyTorch tensor, got {type(tensor_or_array)}")


def _calculate_scaled_event_weights_generalized(
    event_values: Union[np.ndarray, t.Tensor],
    event_original_weights: Union[np.ndarray, t.Tensor],
    bins: np.ndarray,
    total_subtraction_per_bin: Union[np.ndarray, t.Tensor],
) -> Union[np.ndarray, t.Tensor]:
    lib, device = _get_backend_and_device(event_values)
    is_torch = lib == t
    device_kwargs = {"device": device} if is_torch else {}

    raw = _collection(event_values, event_original_weights, total_subtraction_per_bin)

    initial = _collection(
        values=lib.asarray(raw.values, **device_kwargs),
        weights=lib.asarray(raw.weights, **device_kwargs),
        histograms=lib.asarray(raw.histograms, **device_kwargs),
    )

    shape_prefix = _collection(
        values=initial.values.shape[:-1],
        weights=initial.weights.shape[:-1],
        histograms=initial.histograms.shape[:-1],
    )

    bins = lib.asarray(bins, dtype=event_values.dtype, **device_kwargs)
    n_bins, n_events = len(bins) - 1, initial.values.shape[-1]

    flat = _collection(
        initial.values.reshape(-1, n_events).contiguous() if is_torch else initial.values.reshape(-1, n_events),
        initial.weights.reshape(-1, n_events),
        initial.histograms.reshape(-1, n_bins),
    )
    batch_size = _collection(
        values=flat.values.shape[0],
        weights=flat.weights.shape[0],
        histograms=flat.histograms.shape[0],
    )

    common_prefix_dim = np.broadcast_shapes(*shape_prefix.unrolled)
    max_batch_size = int(np.prod(common_prefix_dim)) if common_prefix_dim else 1

    if batch_size.values == 1 and max_batch_size > 1:
        flat.values = lib.broadcast_to(flat.values, (max_batch_size, n_events))
    if batch_size.weights == 1 and max_batch_size > 1:
        flat.weights = lib.broadcast_to(flat.weights, (max_batch_size, n_events))
    if batch_size.histograms == 1 and max_batch_size > 1:
        flat.histograms = lib.broadcast_to(flat.histograms, (max_batch_size, n_bins))

    _digitize, digitize_kwargs = (lib.bucketize, {"right": False}) if is_torch else (lib.digitize, {})
    raw_indices = _digitize(flat.values, bins, **digitize_kwargs) - 1

    is_out_of_bounds = (raw_indices < 0) | (raw_indices >= n_bins)
    event_bin_indices = lib.clip(raw_indices, 0, n_bins - 1)

    event_weights_for_summation = flat.weights.clone() if is_torch else flat.weights.copy()
    event_weights_for_summation[is_out_of_bounds] = 0.0

    sum_original_weights_per_bin = lib.zeros((max_batch_size, n_bins), dtype=flat.weights.dtype, **device_kwargs)
    if is_torch:
        sum_original_weights_per_bin.scatter_add_(1, event_bin_indices.long(), event_weights_for_summation)
    else:
        for index in range(max_batch_size):
            sum_original_weights_per_bin[index] = lib.bincount(
                event_bin_indices[index], event_weights_for_summation[index], n_bins
            )

    scale_factor_per_bin = lib.ones_like(sum_original_weights_per_bin)
    non_zero_sum_mask = sum_original_weights_per_bin != 0
    scale_factor_per_bin[non_zero_sum_mask] = (
        1.0 - flat.histograms[non_zero_sum_mask] / sum_original_weights_per_bin[non_zero_sum_mask]
    )

    zero_sum_non_zero_subtraction_mask = (sum_original_weights_per_bin == 0) & (flat.histograms != 0)
    scale_factor_per_bin[zero_sum_non_zero_subtraction_mask] = 0.0

    if is_torch:
        scale_factors_for_events = lib.gather(scale_factor_per_bin, dim=1, index=event_bin_indices.long())
    else:
        row_idx_gather = lib.arange(max_batch_size)[:, None]
        scale_factors_for_events = scale_factor_per_bin[row_idx_gather, event_bin_indices]

    corrected_event_weights_flat = flat.weights * scale_factors_for_events
    corrected_event_weights_flat[is_out_of_bounds] = flat.weights[is_out_of_bounds]
    return corrected_event_weights_flat.reshape(initial.weights.shape)


def equi_populated_bins(data: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0, 1, n_bins + 1)
    return np.quantile(np.asarray(data), quantiles)


def mask_region(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    mask_common_1 = df.id_tau_vsJet_VLoose_2 > 0.5
    mask_common_4 = (df.iso_1 > 0.02) & (df.iso_1 < 0.15) if channel == "qcd" else (df.iso_1 > 0.0) & (df.iso_1 < 0.15)
    mask_common_5 = (df.extramuon_veto < 0.5) & (df.extraelec_veto < 0.5)

    if channel == "qcd":
        mask = mask_common_1 & (df.q_1 * df.q_2 > 0) & mask_common_4 & mask_common_5 & (df.mt_1 < 50)
    else:
        mask = mask_common_1 & (df.nbtag == 0) & mask_common_4 & mask_common_5 & (df.mt_1 > 70)

    return df[mask].copy()


def split_even_odd(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold1 = df[df.event_var == 0].reset_index(drop=True)
    fold2 = df[df.event_var == 1].reset_index(drop=True)

    train1, val1 = train_test_split(fold1, test_size=0.5, random_state=SEED)
    train2, val2 = train_test_split(fold2, test_size=0.5, random_state=SEED)
    return train1.reset_index(drop=True), val1.reset_index(drop=True), train2.reset_index(drop=True), val2.reset_index(drop=True)


def build_training_dataset(df: pd.DataFrame, training_var: List[str], channel: str) -> _component_collection:
    if channel == "wjets":
        ss_os_split = _same_sign_opposite_sign_split(
            ss=df[df.SS],
            os=df[(df.OS & (df.Label != 2)) | (df.SS & (df.Label == 2))],
        )
    else:
        ss_os_split = _same_sign_opposite_sign_split(
            ss=df[df.SS],
            os=[],
        )

    return _component_collection(
        X=ss_os_split.apply_func(lambda x: x[training_var].to_numpy(dtype=np.float32)),
        Y=ss_os_split.apply_func(lambda x: x["Label"].to_numpy(dtype=np.float32)),
        weights=ss_os_split.apply_func(lambda x: x["weight"].to_numpy(dtype=np.float32)),
        class_weights=ss_os_split.apply_func(lambda x: x["class_weights"].to_numpy()),
        process=ss_os_split.apply_func(lambda x: x["process"].to_numpy(dtype=np.float32)),
        Label=ss_os_split.apply_func(lambda x: x["Label"].to_numpy(dtype=np.float32)),
    )


def build_data_only_dataset(df: pd.DataFrame, training_var: List[str]) -> _component_collection:
    ss_os_split = _same_sign_opposite_sign_split(
        ss=df[df.SS],
        os=df[(df.OS) & (df.process == 0)],
    )
    return _component_collection(
        X=ss_os_split.apply_func(lambda x: x[training_var].to_numpy(dtype=np.float32)),
        Y=ss_os_split.apply_func(lambda x: x["Label"].to_numpy(dtype=np.float32)),
        weights=ss_os_split.apply_func(lambda x: x["weight"].to_numpy(dtype=np.float32)),
        class_weights=ss_os_split.apply_func(lambda x: x["class_weights"].to_numpy()),
        process=ss_os_split.apply_func(lambda x: x["process"].to_numpy(dtype=np.float32)),
        Label=ss_os_split.apply_func(lambda x: x["Label"].to_numpy(dtype=np.float32)),
    )


def _concat_predictions(parts: List[torch.Tensor]) -> np.ndarray:
    if not parts:
        return np.array([], dtype=np.float32)
    return torch.concat(parts, dim=0).detach().cpu().numpy()


def _collect_processwise_probs_weights(
    model1: nn.Module,
    model2: nn.Module,
    train1: _component_collection,
    val1: _component_collection,
    train2: _component_collection,
    val2: _component_collection,
    process_ids: List[int],
    device: torch.device,
    region: Literal["ss", "os"],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    probs_by_process: Dict[int, np.ndarray] = {}
    weights_by_process: Dict[int, np.ndarray] = {}

    model_pairs = ((model1, train2, val2), (model2, train1, val1))
    for process_id in process_ids:
        prob_parts = []
        weight_parts = []
        for model, train_ds, val_ds in model_pairs:
            for ds in (train_ds, val_ds):
                region_process = getattr(ds.process, region)
                mask = region_process == process_id
                if mask.sum() == 0:
                    continue
                region_X = getattr(ds.X, region)
                region_weights = getattr(ds.weights, region)
                prob_parts.append(predict_probabilities(model, region_X[mask], device))
                weight_parts.append(region_weights[mask])

        probs_by_process[process_id] = _concat_predictions(prob_parts)
        weights_by_process[process_id] = _concat_predictions(weight_parts)

    return probs_by_process, weights_by_process


def _collect_wjets_data_probs(
    model1: nn.Module,
    model2: nn.Module,
    train1_data: _component_collection,
    val1_data: _component_collection,
    train2_data: _component_collection,
    val2_data: _component_collection,
    device: torch.device,
) -> np.ndarray:
    return _concat_predictions(
        [
            predict_probabilities(model1, train1_data.X.os, device),
            predict_probabilities(model1, val1_data.X.os, device),
            predict_probabilities(model2, train2_data.X.os, device),
            predict_probabilities(model2, val2_data.X.os, device),
        ]
    )


def _load_saved_process_weights(channel: str, fold1_dir: Path, fold2_dir: Path) -> np.ndarray:
    train_name, val_name = CHANNEL_CONFIG[channel]["saved_weight_files"]
    return torch.concat(
        [
            torch.load(fold2_dir / train_name),
            torch.load(fold2_dir / val_name),
            torch.load(fold1_dir / train_name),
            torch.load(fold1_dir / val_name),
        ],
        dim=0,
    ).detach().cpu().numpy()


def _sum_histograms(values_and_weights: List[Tuple[np.ndarray, np.ndarray]], bins: np.ndarray) -> np.ndarray:
    total = np.zeros(len(bins) - 1, dtype=np.float64)
    for values, weights in values_and_weights:
        if len(values) == 0:
            continue
        total += np.histogram(values, bins=bins, weights=weights)[0]
    return total


def _combine_process_group(
    probs_by_process: Dict[int, np.ndarray],
    weights_by_process: Dict[int, np.ndarray],
    process_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    probs = [probs_by_process[process_id] for process_id in process_ids if len(probs_by_process[process_id]) > 0]
    weights = [weights_by_process[process_id] for process_id in process_ids if len(weights_by_process[process_id]) > 0]
    probs_out = np.concatenate(probs, axis=0) if probs else np.array([], dtype=np.float32)
    weights_out = np.concatenate(weights, axis=0) if weights else np.array([], dtype=np.float32)
    return probs_out, weights_out


def CMS_CHANNEL_TITLE(ax, *args, **kwargs):
    ax[0].set_title(r"$e\tau_h$", fontsize=20, loc="left", fontproperties="Tex Gyre Heros")


def CMS_LUMI_TITLE(ax, *args, **kwargs):
    ax[0].set_title("59.8 $fb^{-1}$ (2018, 13 TeV)", fontsize=20, loc="right", fontproperties="Tex Gyre Heros")


def CMS_LABEL(ax, *args, **kwargs):
    ax[0].text(
        0.025,
        0.95,
        "Private work (CMS data/simulation)",
        fontsize=20,
        verticalalignment="top",
        fontproperties="Tex Gyre Heros:italic",
        bbox=dict(facecolor="white", alpha=0, edgecolor="white", boxstyle="round,pad=0.5"),
        transform=ax[0].transAxes,
    )


def adjust_ylim_for_legend(ax=None, spacing=0.05):
    if ax is None:
        ax = plt.gca()

    fig = ax.figure
    fig.canvas.draw()

    leg = ax.get_legend()
    if leg is None:
        return

    bbox_leg, bbox_ax = leg.get_window_extent(), ax.get_window_extent()
    legend_height_ratio = bbox_leg.height / bbox_ax.height

    ymin, ymax = ax.get_ylim()
    scale = ax.get_yscale()
    available_fraction = max(1.0 - legend_height_ratio - spacing, 0.1)

    if scale == "linear":
        data_max_y = ax.dataLim.y1
        data_range = data_max_y - ymin
        new_ymax = ymin + data_range / available_fraction
        ax.set_ylim(ymin, new_ymax)
    elif scale == "log":
        log_ymin = np.log10(ymin)
        log_data_max = np.log10(ax.dataLim.y1)
        new_log_range = (log_data_max - log_ymin) / available_fraction
        new_ymax = 10 ** np.ceil(log_ymin + new_log_range)
        ax.set_ylim(ymin, new_ymax)


def plot_reduced_data_figure(
    bin_centers: np.ndarray,
    bin_widths: np.ndarray,
    reduced_counts: np.ndarray,
    target_counts: np.ndarray,
    data_counts: np.ndarray,
    target_stat_var: np.ndarray,
    output_dir: Path,
    prefix: str,
    target_label: str,
):
    x_error = 0.5 * bin_widths
    y_error = np.sqrt(data_counts)
    y_error_stat = np.sqrt(target_stat_var)

    fig, ax = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    CMS_CHANNEL_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_LABEL(ax)

    ax[0].errorbar(bin_centers, reduced_counts, yerr=y_error, xerr=x_error, color="black", fmt="o", markersize=5, label="data (reduced)")
    ax[0].bar(bin_centers, target_counts, width=bin_widths, color="#e76300" if prefix == "wjets" else "#b9ac70", label=target_label)
    ax[0].set_ylabel("Events")
    ax[0].legend()
    adjust_ylim_for_legend(ax[0])

    ratio_den = target_counts + 1e-10
    ax[1].errorbar(bin_centers, reduced_counts / ratio_den, yerr=y_error / np.maximum(data_counts, 1.0), xerr=x_error, label="Ratio", color="black", fmt="o")
    ax[1].fill_between(
        bin_centers,
        1 - y_error_stat / ratio_den,
        1 + y_error_stat / ratio_den,
        color="gray",
        alpha=0.3,
        step="mid",
        label="Stat. Unc.",
    )
    ax[1].axhline(1, color="red", linestyle="--", linewidth=1.5)
    ax[1].set_ylabel("Data / Model")
    ax[1].set_ylim([0.5, 1.5])
    ax[1].legend()
    ax[1].set_xlabel("NN output")

    fig.savefig(output_dir / f"results_data_reduced_{prefix}.png")
    fig.savefig(output_dir / f"results_data_reduced_{prefix}.pdf")
    plt.close(fig)


def plot_summary_figure(
    probs_compact: List[np.ndarray],
    weights_compact: List[np.ndarray],
    labels_compact: List[str],
    colors_compact: List[str],
    process_fraction_components: List[np.ndarray],
    bin_centers: np.ndarray,
    bin_widths: np.ndarray,
    bins: np.ndarray,
    data_counts: np.ndarray,
    sim_counts: np.ndarray,
    stat_var: np.ndarray,
    output_dir: Path,
    prefix: str,
    bins_label: str,
    plot_process_fractions: bool,
):
    nrows = 3 if plot_process_fractions else 2
    height_ratios = [3, 1, 1] if plot_process_fractions else [3, 1]
    fig, ax = plt.subplots(nrows, 1, figsize=(12, 12), sharex=True, gridspec_kw={"height_ratios": height_ratios, "hspace": 0.05})
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    CMS_CHANNEL_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_LABEL(ax)

    x_error = 0.5 * bin_widths
    y_error = np.sqrt(data_counts)
    y_err_stat = np.sqrt(stat_var)

    ax[0].hist(probs_compact, bins=bins, weights=weights_compact, histtype="barstacked", label=labels_compact, color=colors_compact)
    ax[0].errorbar(bin_centers, data_counts, yerr=y_error, xerr=x_error, fmt="o", color="black", label="data", markersize=5)
    ax[0].set_ylabel("Events")
    ax[0].set_ylim([0, 1.4 * np.max([np.max(data_counts), np.max(sim_counts)])])
    ax[0].legend(loc="upper right", bbox_to_anchor=(0.8, 0.9), ncol=3, frameon=False)
    ax[0].tick_params(direction="in", top=True, right=True)

    ratio = np.divide(data_counts, sim_counts + 1e-10)
    ax[1].errorbar(bin_centers, ratio, xerr=x_error, yerr=y_error / np.maximum(data_counts, 1.0), fmt="o", color="black", markersize=5, label="ratio")
    ax[1].fill_between(
        bin_centers,
        1 - y_err_stat / (sim_counts + 1e-10),
        1 + y_err_stat / (sim_counts + 1e-10),
        color="gray",
        alpha=0.3,
        step="mid",
        label="Stat. Unc.",
    )
    ax[1].axhline(1, color="red", linestyle="--", linewidth=1.5)
    ax[1].set_ylabel("Data / Sim")
    ax[1].set_ylim([0.5, 1.5])
    ax[1].grid(True, linestyle=":", alpha=0.7)
    ax[1].tick_params(direction="in", top=True, right=True)
    ax[1].legend(loc="upper right", ncol=2)

    if plot_process_fractions:
        running_bottom = np.zeros_like(bin_centers, dtype=np.float64)
        for counts, color in process_fraction_components:
            frac = np.divide(counts, sim_counts + 1e-10)
            ax[2].bar(bin_centers, frac, bottom=running_bottom, color=color, width=bin_widths)
            running_bottom += frac
        ax[2].set_ylabel("Proc. frac.")
        ax[2].set_xlabel("NN output")
    else:
        ax[1].set_xlabel("NN output")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    fig.savefig(output_dir / f"results_training_{bins_label}_{prefix}.png")
    fig.savefig(output_dir / f"results_training_{bins_label}_{prefix}.pdf")
    plt.close(fig)


def main() -> None:
    args = Args().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_complete_path = resolve_path(args.data_complete_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 1.5,
            "axes.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )

    channels = ["wjets", "qcd"] if args.channel == "both" else [args.channel]

    for channel in channels:
        channel_cfg = CHANNEL_CONFIG[channel]
        ckpt_fold1 = resolve_path(args.ckpt_pth_fold1 or channel_cfg["default_ckpt_fold1"])
        ckpt_fold2 = resolve_path(args.ckpt_pth_fold2 or channel_cfg["default_ckpt_fold2"])

        logger.info("Loading %s models", channel)
        model1 = load_model(channel_cfg["input_dim"], ckpt_fold1 / "model_checkpoint.pth", device)
        model2 = load_model(channel_cfg["input_dim"], ckpt_fold2 / "model_checkpoint.pth", device)

        logger.info("Loading data from %s", data_complete_path)
        data_complete = pd.read_feather(data_complete_path)
        data_region = mask_region(data_complete, channel)
        train1_df, val1_df, train2_df, val2_df = split_even_odd(data_region)

        variables = channel_cfg["variables"]
        train1 = build_training_dataset(train1_df, variables, channel).to_torch(device=None)
        val1 = build_training_dataset(val1_df, variables, channel).to_torch(device=None)
        train2 = build_training_dataset(train2_df, variables, channel).to_torch(device=None)
        val2 = build_training_dataset(val2_df, variables, channel).to_torch(device=None)

        probs_by_process, weights_by_process = _collect_processwise_probs_weights(
            model1=model1,
            model2=model2,
            train1=train1,
            val1=val1,
            train2=train2,
            val2=val2,
            process_ids=channel_cfg["process_order"],
            device=device,
            region=channel_cfg["model_region"],
        )

        saved_process_id = channel_cfg["saved_weight_process"]
        saved_process_weights = _load_saved_process_weights(channel, ckpt_fold1, ckpt_fold2)

        if channel == "wjets":
            train1_data = build_data_only_dataset(train1_df, variables).to_torch(device=None)
            val1_data = build_data_only_dataset(val1_df, variables).to_torch(device=None)
            train2_data = build_data_only_dataset(train2_df, variables).to_torch(device=None)
            val2_data = build_data_only_dataset(val2_df, variables).to_torch(device=None)
            probs_data = _collect_wjets_data_probs(model1, model2, train1_data, val1_data, train2_data, val2_data, device)
        else:
            probs_data = probs_by_process[0]

        if args.bins == "equi_populated":
            bins = equi_populated_bins(probs_data, args.n_bins)
        else:
            bins = np.linspace(0, 1, args.n_bins + 1)
        bin_widths = np.diff(bins)

        if len(saved_process_weights) != len(probs_by_process[saved_process_id]):
            if channel == "qcd":
                message = (
                    "Saved QCD template weights from training_qcd.py do not match the current dataset. "
                    "The plotted QCD component must come from qcd_weights_qcd_train.pt / qcd_weights_qcd_val.pt, "
                    "while the write-back column weight_qcd is a separate reduced-data product. "
                    "Please rerun training_qcd.py or point --ckpt_pth_fold1/--ckpt_pth_fold2 to a matching QCD run."
                )
                if args.channel == "both":
                    logger.error("%s Skipping qcd plots for this run.", message)
                    continue
                raise ValueError(message)
            else:
                message = (
                    "Saved Wjets QCD template weights do not match the current dataset. "
                    "These weights must come from qcd_weights_train.pt / qcd_weights_val.pt produced by training_wjets.py, "
                    "and they are distinct from the weight_qcd column written by the QCD workflow. "
                    "Please rerun training_wjets.py or point --ckpt_pth_fold1/--ckpt_pth_fold2 to a matching run."
                )
                if args.channel == "both":
                    logger.error("%s Skipping wjets plots for this run.", message)
                    continue
                raise ValueError(message)

        if len(saved_process_weights) != len(probs_by_process[saved_process_id]):
            raise ValueError(
                f"Aligned saved weights for process {saved_process_id} still have length {len(saved_process_weights)}, "
                f"but predicted probabilities have length {len(probs_by_process[saved_process_id])}."
            )

        plot_weights_by_process = {process_id: weights.copy() for process_id, weights in weights_by_process.items()}
        plot_weights_by_process[saved_process_id] = saved_process_weights

        target_process = channel_cfg["target_process"]

        if channel == "wjets":
            background_ids = [process_id for process_id in channel_cfg["process_order"] if process_id not in (target_process, saved_process_id)]
            subtraction_terms = [
                (probs_by_process[process_id], plot_weights_by_process[process_id])
                for process_id in background_ids
            ]
            subtraction_terms.append((probs_by_process[saved_process_id], plot_weights_by_process[saved_process_id]))
        else:
            background_ids = [process_id for process_id in channel_cfg["process_order"] if process_id != target_process]
            subtraction_terms = [
                (probs_by_process[process_id], plot_weights_by_process[process_id])
                for process_id in background_ids
            ]

        background_hist = _sum_histograms(subtraction_terms, bins)
        reduced_weights = _calculate_scaled_event_weights_generalized(
            event_values=probs_data,
            event_original_weights=np.ones_like(probs_data),
            bins=bins,
            total_subtraction_per_bin=background_hist,
        )

        if args.write_back:
            if channel == "wjets":
                write_mask = (data_region["OS"] == True) & (data_region["process"] == 0)
            else:
                write_mask = (data_region["SS"] == True) & (data_region["process"] == 0)

            indices = data_region.index[write_mask].to_numpy()
            if len(indices) != len(reduced_weights):
                raise ValueError(
                    f"Mask selects {len(indices)} rows but computed weights have length {len(reduced_weights)}."
                )

            column_name = channel_cfg["saved_weight_column"]
            data_region[column_name] = np.nan
            data_region.loc[indices, column_name] = reduced_weights

            data_complete[column_name] = np.nan
            data_complete.loc[data_region.index, column_name] = data_region[column_name]
            data_complete.reset_index(drop=True).to_feather(data_complete_path)
            logger.info("Successfully inserted %s into %s", column_name, data_complete_path)
        else:
            logger.info("Skipping write-back to data frame/file (--write_back is False).")

        data_counts, bin_edges = np.histogram(probs_data, bins=bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        reduced_counts, _ = np.histogram(probs_data, weights=reduced_weights, bins=bins)

        process_counts = {
            process_id: np.histogram(probs_by_process[process_id], weights=plot_weights_by_process[process_id], bins=bins)[0]
            for process_id in channel_cfg["process_order"]
        }
        process_vars = {
            process_id: np.histogram(probs_by_process[process_id], weights=plot_weights_by_process[process_id] ** 2, bins=bins)[0]
            for process_id in channel_cfg["process_order"]
        }

        sim_counts = np.zeros(len(bins) - 1, dtype=np.float64)
        for process_id in channel_cfg["process_order"]:
            sim_counts += process_counts[process_id]

        target_counts = process_counts[target_process]
        target_var = process_vars[target_process]
        total_stat_var = np.zeros(len(bins) - 1, dtype=np.float64)
        for process_id in channel_cfg["process_order"]:
            total_stat_var += process_vars[process_id]

        compact_probs = []
        compact_weights = []
        compact_labels = []
        compact_colors = []
        fraction_components = []

        for label, process_ids, color in channel_cfg["compact_groups"]:
            group_probs, group_weights = _combine_process_group(probs_by_process, plot_weights_by_process, process_ids)
            compact_probs.append(group_probs)
            compact_weights.append(group_weights)
            compact_labels.append(label)
            compact_colors.append(color)
            group_counts = np.zeros(len(bins) - 1, dtype=np.float64)
            for process_id in process_ids:
                group_counts += process_counts[process_id]
            fraction_components.append((group_counts, color))

        plot_reduced_data_figure(
            bin_centers=bin_centers,
            bin_widths=bin_widths,
            reduced_counts=reduced_counts,
            target_counts=target_counts,
            data_counts=data_counts,
            target_stat_var=target_var,
            output_dir=output_dir,
            prefix=channel_cfg["output_prefix"],
            target_label=channel_cfg["reduced_plot_label"],
        )

        plot_summary_figure(
            probs_compact=compact_probs,
            weights_compact=compact_weights,
            labels_compact=compact_labels,
            colors_compact=compact_colors,
            process_fraction_components=fraction_components,
            bin_centers=bin_centers,
            bin_widths=bin_widths,
            bins=bins,
            data_counts=data_counts,
            sim_counts=sim_counts,
            stat_var=total_stat_var,
            output_dir=output_dir,
            prefix=channel_cfg["output_prefix"],
            bins_label="equi" if args.bins == "equi_populated" else "uniform",
            plot_process_fractions=args.plot_process_fractions,
        )


if __name__ == "__main__":
    main()
