'''
loads models, evaluates fake factors, and makes plots
'''

from __future__ import annotations
import logging
import hashlib
from contextlib import contextmanager
from copy import deepcopy
import correctionlib as cr
from pathlib import Path
import os
import random 
import re
from typing import Literal, Tuple, Iterable

import matplotlib.pyplot as plt
import matplotlib
import mplhep as hep
import numpy as np
import pandas as pd
from tap import Tap
from tayloranalysis import extend_model as _ta_extend
import torch
import torch as t
import yaml

from classes.Logging import setup_logging
from classes.NeuralNetworks import RealNVP
from classes.Dataclasses import ModelConfig
from classes.Collection import load_model_config, load_flow, load_conditional_flow, evaluate_pdf, compute_eventwise_fake_factors, get_my_data_qcd
from classes.Collection import load_config, load_grouped_qcd_njets_router
from classes.Plotting import CMS_CHANNEL_TITLE, CMS_LABEL, CMS_LUMI_TITLE, CMS_NJETS_TITLE, add_cms_privatework_lumi_row, reorder_for_rowwise_legend, adjust_ylim_for_legend
from classes.NF import corr_matrix_nfsample_data

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

logger = setup_logging(logger=logging.getLogger(__name__))

SEED = 42

class Args(Tap):
    model_mode: Literal['grouped_njets_split', 'single_nf', 'conditional_nf'] = 'conditional_nf'  # Training mode to load: grouped NF split by njets, single inclusive NF, or conditional NF with njets as input.
    classifier_training_tag: str = ''  # Optional classifier training folder suffix after 'training_'. Empty -> pick most recent.
    classifier_hidden_layers: int = 2  # Binary-classifier selection helper: pick the most recent training with this number of hidden layers.
    plot_training_diagnostics: bool = True   # Plot training loss / learning-rate / time-per-epoch curves.
    plot_nf_sampling: bool = True           # Plot NF-sampled vs data histograms in training variables.
    plot_ff_results: bool = True             # Plot fake-factor comparison stacks for each njets category.
    plot_ff_values: bool = True              # Plot FF values in histogram
    plot_ar_data_with_clipping: bool = True  # Plot AR data with both kept and excluded events (by clipping mask).
    plot_taylor_coefficients: bool = True   # Compute and plot first-order Taylor coefficients (mean |d log p/d x_i|). Slow — needs a backward pass.
    plot_complete_variables: bool = True
    ratio_ylim_min: float = 0.5  # Lower y-limit for ratio panels.
    ratio_ylim_max: float = 1.5  # Upper y-limit for ratio panels.

    taus = 1 #[1, 2] #[1, 2, 12] # list of tau fakes
    embedding: Literal["embedding", "no_embedding"] = "embedding"
    var = "variables_61"


# Runtime context (initialized in `initialize_runtime_context()` and consumed by plotting functions)
cfg_path = load_config('/work/tapp/TauFF/NF4FF/Normalizing_Flow_tt/configs/config_path.yaml')
args = Args().parse_args()

variables = []
dim = 0
training_variables_tag = ''
variables_with_njets = []
device = None
config_path = cfg_path['config_NF']
mode_dir = ''
include_njets_feature = False
resolved_tag = ''

data_complete = None
list_variables = []
labels = {}
labels_short = {}
list_xlabels = []
list_bins = []
main_plot_bins_by_variable = {}
sampling_plot_bins_by_variable = {}
plot_root_dir = Path('plots')
MASKS_CONFIG_PATH = Path('/work/tapp/TauFF/NF4FF/Normalizing_Flow_tt/configs/masks.yaml')
MASKS_CONFIG: dict[str, list[str]] = {}

# ------------ functions ----------


def reserve_cms_label_space(ax, factor=5.0):
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * factor)


def equi_populated_bins(data, n_bins):

    data = np.asarray(data)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(data, quantiles)
    return bin_edges


def should_log_plot_progress(index: int, total: int, step: int = 5) -> bool:
    return index == 1 or index == total or index % step == 0


def draw_stacked_stepfill(ax, bin_edges, components: list[tuple[np.ndarray, str, str]]) -> np.ndarray:
    cumulative = np.zeros(len(bin_edges) - 1, dtype=float)
    final_top = cumulative.copy()

    for counts, color, label in components:
        next_cumulative = cumulative + counts
        ax.fill_between(
            bin_edges,
            np.r_[cumulative, cumulative[-1]],
            np.r_[next_cumulative, next_cumulative[-1]],
            step='post',
            color=color,
            linewidth=0,
            label=label,
        )
        ax.stairs(next_cumulative, bin_edges, color='black', linewidth=1.0)
        cumulative = next_cumulative
        final_top = next_cumulative

    return final_top


def build_training_variables_tag(variables: list[str]) -> str:
    variables_joined = "|".join(variables)
    variables_hash = hashlib.sha1(variables_joined.encode("utf-8")).hexdigest()[:8]
    tail_variables = variables[4:]
    if tail_variables:
        readable_tail = "_".join(tail_variables)
        readable_tail = re.sub(r"[^A-Za-z0-9_]+", "_", readable_tail).strip("_")
    else:
        readable_tail = "none"
    
    if 'deltaR_ditaupair' in variables and 'pt_1' in variables:
        return f"vars{len(variables)}_{readable_tail}_{variables_hash}"
    elif 'pt_1' not in variables:
        return f"vars{len(variables)}2_{readable_tail}_{variables_hash}"
    elif 'deltaR_ditaupair' not in variables and 'pt_1' in variables: 
        return f"vars{len(variables)}1_{readable_tail}_{variables_hash}"
    else:
        return f"vars{len(variables)}a_{readable_tail}_{variables_hash}"


def _build_training_variables_prefix(variables: list[str]) -> str:
    """Return the hash-free readable prefix, e.g. 'vars5_pt_vis'."""
    tail_variables = variables[4:]
    if tail_variables:
        readable_tail = "_".join(tail_variables)
        readable_tail = re.sub(r"[^A-Za-z0-9_]+", "_", readable_tail).strip("_")
    else:
        readable_tail = "none"

    if 'deltaR_ditaupair' in variables and 'pt_1' in variables:
        return f"vars{len(variables)}_{readable_tail}"
    elif 'pt_1' not in variables:
        return f"vars{len(variables)}2_{readable_tail}"
    elif 'deltaR_ditaupair' not in variables and 'pt_1' in variables:
        return f"vars{len(variables)}1_{readable_tail}"
    else:
        return f"vars{len(variables)}a_{readable_tail}"


def resolve_training_tag(variables: list[str], mode_dir: str, base_dir: str = cfg_path['NF_results']) -> str:
    """
    Glob for a training folder whose name starts with 'training_<prefix>'
    (ignoring the trailing hash).  Returns the folder-name suffix that follows
    'training_', so the caller can use it identically to training_variables_tag.

    If exactly one matching folder is found it is used.  If several match the
    most-recently-modified one is chosen.  If none match, fall back to the
    exact computed tag (old behaviour).
    """
    exact_tag = build_training_variables_tag(variables)
    prefix = _build_training_variables_prefix(variables)
    search_root = Path(base_dir) / mode_dir
    if not search_root.exists():
        logger.warning('Training base dir not found: %s — falling back to exact tag', search_root)
        return exact_tag

    candidates = sorted(search_root.glob(f'training_{prefix}*'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        logger.warning(
            'No training folder matching training_%s* found in %s — falling back to exact tag %s',
            prefix, search_root, exact_tag,
        )
        return exact_tag
    if len(candidates) > 1:
        logger.warning(
            'Multiple training folders match training_%s* in %s: %s — using most recent: %s',
            prefix, search_root,
            [c.name for c in candidates],
            candidates[0].name,
        )
    chosen = candidates[0].name  # e.g. 'training_vars5_pt_vis_137f51a0'
    resolved_tag = chosen.removeprefix('training_')
    logger.info('Resolved training tag: %s -> %s', exact_tag, resolved_tag)
    return resolved_tag


def resolve_latest_training_tag(base_dir: str, prefix: str = 'training_') -> str:
    search_root = Path(base_dir)
    if not search_root.exists():
        raise FileNotFoundError(f'Training base dir not found: {search_root}')
    candidates = sorted(search_root.glob(f'{prefix}*'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f'No {prefix}* folders found in {search_root}')
    return candidates[0].name.removeprefix('training_')


def correction_classifier_paths(resolved_tag: str) -> tuple[str, str]:
    base = f'Training_results_new/binary_classifier_corrections/training_{resolved_tag}/Wjets'
    return (
        f'{base}/DR/SR_AR_classifier/latest',
        f'{base}/antiDR/SR_AR_classifier/latest',
    )


def _normalize_learning_rate(value) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return np.nan
        return float(value[0])
    return float(value)


def _slugify_plot_label(label: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_').lower()


def evaluate_compound_ff_correction(correction_set, compound_name: str, df: pd.DataFrame) -> np.ndarray:
    compound_correction = correction_set.compound[compound_name]
    expected_inputs = [input_spec.name for input_spec in compound_correction.inputs]

    input_values = {
        'tau_decaymode_2': df.tau_decaymode_2,
        'eta_1': df.eta_1,
        'eta_2': df.eta_2,
        'jeta_1': df.jeta_1,
        'jeta_2': df.jeta_2,
        'jpt_1': df.jpt_1,
        'jpt_2': df.jpt_2,
        'met': df.met,
        'deltaR_ditaupair': df.deltaR_ditaupair,
        'deltaR_1j1': df.deltaR_1j1,
        'deltaR_12j1': df.deltaR_12j1,
        'pt_ttjj': df.pt_ttjj,
        'mass_2': df.mass_2,
        'mt_tot': df.mt_tot,
        'm_vis': df.m_vis,
        'iso_1': df.iso_1,
        'njets': df.njets,
        'syst': 'nominal',
    }

    missing_inputs = [name for name in expected_inputs if name not in input_values]
    if missing_inputs:
        raise KeyError(f'Missing input mapping for correction {compound_name}: {missing_inputs}')

    ordered_inputs = [input_values[name] for name in expected_inputs]
    return compound_correction.evaluate(*ordered_inputs)


def load_saved_model_config(checkpoint_dir: str | Path, fallback_path: str | Path) -> ModelConfig:
    saved_config_path = Path(checkpoint_dir).parent / 'config.yaml'
    if saved_config_path.exists():
        with open(saved_config_path, 'r') as handle:
            raw = yaml.unsafe_load(handle)

        if isinstance(raw, ModelConfig):
            return raw

        values = vars(raw) if hasattr(raw, '__dict__') else raw
        return ModelConfig(
            n_layers=values['n_layers'],
            hidden_dims=values['hidden_dims'],
            s_scale=values['s_scale'],
            use_cut_preprocessing=values.get('use_cut_preprocessing', True),
            cut_preprocessing_index=values.get('cut_preprocessing_index', [0, 1]),
            cut_preprocessing_thresholds=values.get('cut_preprocessing_thresholds', [33.0, 30.0]),
            cut_preprocessing_epsilon=values.get('cut_preprocessing_epsilon', 1e-6),
            use_tail_preprocessing=values.get('use_tail_preprocessing', False),
            tail_preprocessing_index=values.get('tail_preprocessing_index', 2),
            tail_preprocessing_type=values.get('tail_preprocessing_type', 'asinh'),
            tail_preprocessing_center=values.get('tail_preprocessing_center', 0.0),
            tail_preprocessing_scale=values.get('tail_preprocessing_scale', 1.0),
            tail_preprocessing_epsilon=values.get('tail_preprocessing_epsilon', 1e-6),
        )

    logger.warning('Saved model config not found for %s; falling back to %s', checkpoint_dir, fallback_path)
    return load_model_config(str(fallback_path))


def load_training_history(checkpoint_dir: str | Path) -> pd.DataFrame | None:
    history_path = Path(checkpoint_dir) / 'training_logs.pkl'
    if not history_path.exists():
        logger.warning('Training history not found: %s', history_path)
        return None

    history = pd.read_pickle(history_path)
    if history.empty:
        logger.warning('Training history is empty: %s', history_path)
        return None

    history = history.copy()
    if 'type' in history.columns:
        history = history[history['type'] == 'epoch'].copy()

    required_columns = {'epoch', 'train_loss', 'val_loss', 'lr'}
    missing_columns = required_columns.difference(history.columns)
    if missing_columns:
        logger.warning('Training history %s misses columns: %s', history_path, ', '.join(sorted(missing_columns)))
        return None

    history['epoch'] = pd.to_numeric(history['epoch'], errors='coerce')
    history['train_loss'] = pd.to_numeric(history['train_loss'], errors='coerce')
    history['val_loss'] = pd.to_numeric(history['val_loss'], errors='coerce')
    history['lr'] = history['lr'].apply(_normalize_learning_rate)
    history = history.dropna(subset=['epoch', 'train_loss', 'val_loss', 'lr']).sort_values('epoch')

    if history.empty:
        logger.warning('Training history has no plottable rows: %s', history_path)
        return None

    return history


def plot_training_history_axis(axis, history: pd.DataFrame, label: str) -> tuple[list, list]:
    epochs = history['epoch'].to_numpy()
    train_loss = history['train_loss'].to_numpy()
    val_loss = history['val_loss'].to_numpy()
    learning_rate = history['lr'].to_numpy()

    has_time = 'time_s' in history.columns and history['time_s'].notna().any()

    axis.set_title(label, fontsize=18, loc='center', fontproperties='Tex Gyre Heros')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Loss')
    axis.grid(True, which='major', linestyle=':', alpha=0.35)

    train_line, = axis.plot(
        epochs,
        train_loss,
        color='#1f77b4',
        linewidth=1.9,
        label='Train loss',
    )
    val_line, = axis.plot(
        epochs,
        val_loss,
        color='#d62728',
        linewidth=1.9,
        label='Validation loss',
    )

    lr_axis = axis.twinx()
    lr_line, = lr_axis.plot(
        epochs,
        learning_rate,
        color='black',
        linestyle='--',
        linewidth=1.4,
        label='Learning rate',
    )
    if np.all(learning_rate > 0):
        lr_axis.set_yscale('log')
    lr_axis.set_ylabel('Learning rate')
    lr_axis.grid(False)

    axis.tick_params(direction='in', top=True, right=False)
    lr_axis.tick_params(direction='in', top=True, right=True)

    handles = [train_line, val_line, lr_line]
    labels = [h.get_label() for h in handles]

    if has_time:
        time_s = history['time_s'].to_numpy()
        time_axis = axis.twinx()
        # Push the second right-hand spine outward so it does not overlap the LR axis.
        time_axis.spines['right'].set_position(('axes', 1.18))
        time_line, = time_axis.plot(
            epochs,
            time_s,
            color='#2ca02c',
            linestyle=':',
            linewidth=1.4,
            alpha=0.75,
            label='Time / epoch (s)',
        )
        time_axis.set_ylabel('Time per epoch (s)', color='#2ca02c')
        time_axis.tick_params(axis='y', colors='#2ca02c', direction='in')
        time_axis.grid(False)
        handles.append(time_line)
        labels.append(time_line.get_label())

    return handles, labels


def plot_training_histories(log_specs: list[tuple[str, str | Path]], output_dir: Path) -> None:
    histories: list[tuple[str, pd.DataFrame]] = []
    for label, checkpoint_dir in log_specs:
        history = load_training_history(checkpoint_dir)
        if history is not None:
            histories.append((label, history))

    if not histories:
        logger.warning('No training histories available for plotting.')
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    ncols = 2
    nrows = int(np.ceil(len(histories) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.0 * ncols, 5.2 * nrows), squeeze=False)
    fig.subplots_adjust(top=0.83, hspace=0.34, wspace=0.45)

    flat_axes = axes.flatten()
    legend_handles = None
    legend_labels = None

    for axis, (label, history) in zip(flat_axes, histories):
        handles, labels = plot_training_history_axis(axis, history, label)
        if legend_handles is None:
            legend_handles = handles
            legend_labels = labels

    for axis in flat_axes[len(histories):]:
        axis.axis('off')

    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.96),
        )

    fig.suptitle('Training history overview', y=0.99, fontproperties='Tex Gyre Heros', fontsize=18)
    fig.savefig(output_dir / 'training_history_overview.png', bbox_inches='tight')
    fig.savefig(output_dir / 'training_history_overview.pdf', bbox_inches='tight')
    plt.close(fig)

    for label, history in histories:
        single_fig, single_axis = plt.subplots(figsize=(9.8, 5.4))
        single_fig.subplots_adjust(top=0.84, right=0.78)
        single_handles, single_labels = plot_training_history_axis(single_axis, history, label)
        single_axis.legend(single_handles, single_labels, loc='upper right', frameon=False)
        output_stub = _slugify_plot_label(label)
        single_fig.savefig(output_dir / f'{output_stub}.png', bbox_inches='tight')
        single_fig.savefig(output_dir / f'{output_stub}.pdf', bbox_inches='tight')
        plt.close(single_fig)

    logger.info('Saved training history overview to %s', output_dir)

def plot_pdf_histogram(
    pdf_SR_like_wjets: np.ndarray,
    pdf_SR_like_qcd: np.ndarray,
    pdf_AR_like_wjets: np.ndarray,
    pdf_AR_like_qcd: np.ndarray,
):
    # ------------------------------------------------------------
    # Log-spaced bins
    # ------------------------------------------------------------
    bins = np.logspace(-6, 0, 61)

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # ------------------------------------------------------------
    # W+jets PDFs
    # ------------------------------------------------------------
    ax.hist(
        pdf_SR_like_wjets,
        bins=bins,
        histtype="step",
        linewidth=1.8,
        label="W+jets SR-like PDF",
        color="#e76300",
    )

    ax.hist(
        pdf_AR_like_wjets,
        bins=bins,
        histtype="step",
        linewidth=1.8,
        linestyle="--",
        label="W+jets AR-like PDF",
        color="#e76300",
    )

    # ------------------------------------------------------------
    # QCD PDFs
    # ------------------------------------------------------------
    ax.hist(
        pdf_SR_like_qcd,
        bins=bins,
        histtype="step",
        linewidth=1.8,
        label="QCD SR-like PDF",
        color="#b9ac70",
    )

    ax.hist(
        pdf_AR_like_qcd,
        bins=bins,
        histtype="step",
        linewidth=1.8,
        linestyle="--",
        label="QCD AR-like PDF",
        color="#b9ac70",
    )

    # ------------------------------------------------------------
    # Axis configuration
    # ------------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-6, 1)
    ax.set_ylim(1e-14, 1e-5)

    ax.set_xlabel("PDF value")
    ax.set_ylabel("Events")
    ax.legend()

    ax.set_title(
        "NF-transformed variable PDFs in SR-like vs AR-like regions",
        pad=20
    )

    # ------------------------------------------------------------
    # CMS styling
    # ------------------------------------------------------------
    CMS_CHANNEL_TITLE([ax])
    CMS_LUMI_TITLE([ax])
    CMS_LABEL([ax])

    return fig, ax


def plot_pdf_distributions(
    model_AR: RealNVP,
    model_SR: RealNVP,
    X_events: np.ndarray,
    plot_dir: str | Path,
    title_suffix: str = "",
    clip_mask: np.ndarray | None = None,
) -> None:
    """
    Plot the AR-like and SR-like PDF distributions for the same events
    in a single axis using unfilled step histograms.

    If a clipping mask is applied, the output filename is suffixed with
    '_clipped' and saved in both PNG and PDF formats.
    """

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Apply clipping mask if provided
    # ---------------------------------------------------------
    clipped = clip_mask is not None
    if clipped:
        clip_mask = np.asarray(clip_mask, dtype=bool)
        if clip_mask.shape[0] != X_events.shape[0]:
            raise ValueError(
                "clip_mask must have the same length as X_events."
            )
        X_events = X_events[clip_mask]

    # ---------------------------------------------------------
    # Convert to tensor
    # ---------------------------------------------------------
    X_t = torch.tensor(X_events, dtype=torch.float32, device=device)

    # ---------------------------------------------------------
    # Evaluate PDFs
    # ---------------------------------------------------------
    pdf_AR = evaluate_pdf(model_AR, X_t)
    pdf_SR = evaluate_pdf(model_SR, X_t)

    # ---------------------------------------------------------
    # Histogram bins (log-spaced)
    # ---------------------------------------------------------
    bins = np.logspace(-30, -4, 100)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        pdf_AR,
        bins=bins,
        histtype="step",
        linewidth=1.8,
        color="#1f77b4",
        label="AR-like PDF",
    )

    ax.hist(
        pdf_SR,
        bins=bins,
        histtype="step",
        linewidth=1.8,
        linestyle="--",
        color="#d62728",
        label="SR-like PDF",
    )

    # ---------------------------------------------------------
    # Axis configuration
    # ---------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(bins[0], bins[-1])

    ax.set_xlabel("PDF value")
    ax.set_ylabel("Events")

    ax.set_title(
        f"AR-like vs SR-like PDF distributions {title_suffix}",
        pad=25,
    )

    ax.legend(frameon=False)

    # ---------------------------------------------------------
    # CMS decorations
    # ---------------------------------------------------------

    reserve_cms_label_space(ax, factor=5.0)

    CMS_CHANNEL_TITLE([ax])
    CMS_LUMI_TITLE([ax])
    CMS_LABEL([ax])
    CMS_NJETS_TITLE([ax], title=r"$\mathrm{N_{jets}} \geq 0$")

    # ---------------------------------------------------------
    # Output filenames
    # ---------------------------------------------------------
    suffix = "_clipped" if clipped else ""
    png_path = plot_dir / f"hist_PDFs{suffix}.png"
    pdf_path = plot_dir / f"hist_PDFs{suffix}.pdf"

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    plt.tight_layout()
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close(fig)


def plot_ff_clipping_histogram(
    ff_full_qcd: np.ndarray,
    clip_mask_qcd: np.ndarray,
    clip_value_qcd: float,
    qcd_clipped_percent: float,
    plot_dir: str | Path,
    njets_title: str,
    tau: str
) -> None:


    """Plot and save the FF clipping diagnostic histogram (`hist_FF.png`)."""
    bins = np.logspace(-3, 1, 61)
    ff_kept_qcd = ff_full_qcd[clip_mask_qcd]
    ff_clipped_qcd = ff_full_qcd[~clip_mask_qcd]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10.5))

    CMS_CHANNEL_TITLE([ax])
    CMS_LUMI_TITLE([ax])
    CMS_LABEL([ax])
    CMS_NJETS_TITLE([ax], title=njets_title)

    ax.hist(ff_kept_qcd, bins=bins, label="QCD FF (kept)", color="#b9ac70", alpha=0.9)
    ax.hist(ff_clipped_qcd, bins=bins, label="QCD FF (clipped)", color="#b9ac70", alpha=0.25)
    ax.axvline(clip_value_qcd, color="black", linestyle="--", linewidth=1.4, label=fr"QCD clip ({clip_value_qcd:.2f})")
    ax.set_xscale("log")
    ax.set_yscale('log')
    ax.set_xlim(1e-3, 1e1)
    ax.set_ylabel("Events")
    ax.set_xlabel("Eventwise FF")
    ax.text(
        0.98,
        0.94,
        f"Clipped: {qcd_clipped_percent:.2f}%",
        transform=ax.transAxes,
        ha='right',
        va='top',
    )
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.2)  # add 20% headroom

   
    handles1, labels1 = ax.get_legend_handles_labels()
    fig.legend(handles1, labels1,
               loc='upper left', #bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f"hist_FF_{tau}.png")
    plt.close(fig)

def total_ff_corrected(df):
    df = df.copy()
    ff = cr.CorrectionSet.from_file('/work/mmoser/TauFakeFactors/workdir/ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz')

    frac = ff['process_fractions']


    ff_qcd = ff['QCD_fake_factors']

    corr = cr.CorrectionSet.from_file('/work/mmoser/TauFakeFactors/workdir/ff_2026_01_19_check_variable/2018/FF_corrections_et.json.gz')

    df['qcd_classic_ff'] = ff_qcd.evaluate(
        df.pt_2.values,
        df.njets.values,
        "nominal",
    )


    df["qcd_corrected_classic_ff"] = df["qcd_classic_ff"] * evaluate_compound_ff_correction(
        corr,
        "QCD_compound_correction",
        df,
    ) * corr["QCD_DR_SR_correction"].evaluate(
        df.pt_tt,
        df.njets,
        "nominal",
    )

    df['process_fraction_qcd'] = frac.evaluate(
        'QCD',
        df.mt_1.values,
        df.njets.values,
        'nominal'
    )

    df['corrected_ff'] = df['process_fraction_qcd'] * df['qcd_corrected_classic_ff']

    return df.copy()

def normalizing_flow_ff(
    df1,
    df2,
    variables,
    model_AR_like_tau1,
    model_SR_like_tau1,
    model_AR_like_tau2,
    model_SR_like_tau2,
    global_ff_tau1,
    global_ff_tau2,
    device,
    plotting=True,
    plot_dir="plots",
    include_njets=True,
    njets_title=None,
):
    """
    Computes eventwise fake factors for W+jets and QCD, and returns
    a single DataFrame with the FFs added as new columns.

    Args:
        df : pandas DataFrame with the input events
        variables : list of NF feature names
        model_* : trained RealNVP models
        global_ff_* : global fake factor normalization
        device : torch device
        plotting : bool, plot histograms of FFs
        plot_dir : output directory for diagnostic NF plots

    Returns:
        df : pandas DataFrame with added columns:
             'ff_nf_wjets', 'ff_nf_qcd'
    """
    df1 = df1.copy()
    if df1.empty:
        df1['ff_nf_tau1'] = pd.Series(dtype=float)
        df1['ff_nf'] = pd.Series(dtype=float)
        return df1
    
    df2 = df2.copy()
    if df2.empty:
        df2['ff_nf_tau2'] = pd.Series(dtype=float)
        df2['ff_nf'] = pd.Series(dtype=float)
        return df2

    input_variables = (['njets'] + variables) if include_njets else list(variables)
    
    # --- FF ---
    # Evaluate tau1 PDFs
    df_pt_tau1 = get_my_data_qcd(df1, input_variables).to_torch().to(device)

    pdf_AR_like_tau1 = evaluate_pdf(model_AR_like_tau1, df_pt_tau1.X)
    pdf_SR_like_tau1 = evaluate_pdf(model_SR_like_tau1, df_pt_tau1.X)

    ff_full_tau1, _, global_ff_cor_tau1, clip_mask_tau1, clip_value_tau1 = compute_eventwise_fake_factors(
        pdf_AR_like_tau1, pdf_SR_like_tau1, global_ff_tau1
    )

    # Evaluate tau2 PDFs
    df_pt_tau2 = get_my_data_qcd(df2, input_variables).to_torch().to(device)

    pdf_AR_like_tau2 = evaluate_pdf(model_AR_like_tau2, df_pt_tau2.X)
    pdf_SR_like_tau2 = evaluate_pdf(model_SR_like_tau2, df_pt_tau2.X)

    ff_full_tau2, _, global_ff_cor_tau2, clip_mask_tau2, clip_value_tau2 = compute_eventwise_fake_factors(
        pdf_AR_like_tau2, pdf_SR_like_tau2, global_ff_tau2
    )

    

    # Keep per-process clipping/correction independent.
    # `compute_eventwise_fake_factors` already applies each process-specific
    # global correction. A second combined correction can strongly over-scale FFs.

    if not np.any(clip_mask_tau1):
        logger.warning("No events survive joint FF clipping; returning empty dataframe.")
        df = df.iloc[0:0].copy()
        #df['ff_nf_tau1'] = pd.Series(dtype=float)
        #df['ff_nf'] = pd.Series(dtype=float)
        return df
    
    if not np.any(clip_mask_tau2):
        logger.warning("No events survive joint FF clipping; returning empty dataframe.")
        df = df.iloc[0:0].copy()
        #df['ff_nf_tau2'] = pd.Series(dtype=float)
        #df['ff_nf'] = pd.Series(dtype=float)
        return df

    logger.info(
        "%s clipping acceptance: tau1=%.4f, tau2=%.4f",
        'NF',
        float(np.mean(clip_mask_tau1)),
        float(np.mean(clip_mask_tau2)),
    )

    tau1_clipped_percent = 100.0 * (1.0 - float(np.mean(clip_mask_tau1)))
    tau2_clipped_percent = 100.0 * (1.0 - float(np.mean(clip_mask_tau2)))

    # ----- Factor 0.5 to avoid over-scaling when combining two taus' FFs in the final product -----
    df1 = df1[clip_mask_tau1].copy()
    df1['ff_nf_tau1'] = ff_full_tau1[clip_mask_tau1]

    df2 = df2[clip_mask_tau2].copy()
    df2['ff_nf_tau2'] = ff_full_tau2[clip_mask_tau2]

    # --- plotting ---
    if plotting:
        plot_ff_clipping_histogram(
            ff_full_qcd=ff_full_tau1,
            clip_mask_qcd=clip_mask_tau1,
            clip_value_qcd=clip_value_tau1,
            qcd_clipped_percent=tau1_clipped_percent,
            plot_dir=plot_dir,
            njets_title=njets_title,
            tau = "tau1")
        
        plot_pdf_distributions(
            model_AR=model_AR_like_tau1,
            model_SR=model_SR_like_tau1,
            X_events=df_pt_tau1.X.cpu().numpy(),
            plot_dir=Path(plot_dir) / "tau1_PDFs",
            title_suffix="(Tau1 features)",
            clip_mask=~clip_mask_tau1,
        )

        plot_ff_clipping_histogram(
            ff_full_qcd=ff_full_tau2,
            clip_mask_qcd=clip_mask_tau2,
            clip_value_qcd=clip_value_tau2,
            qcd_clipped_percent=tau2_clipped_percent,
            plot_dir=plot_dir,
            njets_title=njets_title,
            tau = "tau2")
        
        plot_pdf_distributions(
            model_AR=model_AR_like_tau2,
            model_SR=model_SR_like_tau2,
            X_events=df_pt_tau2.X.cpu().numpy(),
            plot_dir=Path(plot_dir) / "tau2_PDFs",
            title_suffix="(Tau2 features)",
            clip_mask=~clip_mask_tau2,
        )

    # --- assemble combined NF fake factor (process-fraction weighted) ---
    """
    classic FF:
    _ff_file = cr.CorrectionSet.from_file('/work/mmoser/TauFakeFactors/workdir/ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz')
    _corr_file = cr.CorrectionSet.from_file('/work/mmoser/TauFakeFactors/workdir/ff_2026_01_19_check_variable/2018/FF_corrections_et.json.gz')

    _frac = _ff_file['process_fractions']
    _ff_ttbar = _ff_file['ttbar_fake_factors']

    df['ttbar_classic_ff'] = _ff_ttbar.evaluate(
        df.pt_2.values,
        df.njets.values,
        "nominal",
    )
    df['ttbar_corrected_classic_ff'] = df['ttbar_classic_ff'] * evaluate_compound_ff_correction(
        _corr_file,
        'ttbar_compound_correction',
        df,
    )

    df['process_fraction_wjets'] = _frac.evaluate('Wjets', df.mt_1.values, df.njets.values, 'nominal')
    df['process_fraction_qcd'] = _frac.evaluate('QCD', df.mt_1.values, df.njets.values, 'nominal')
    df['process_fraction_ttbar'] = _frac.evaluate('ttbar', df.mt_1.values, df.njets.values, 'nominal')

    df['ff_nf'] = (
        df['process_fraction_wjets'] * df['ff_nf_wjets']
        + df['process_fraction_qcd'] * df['ff_nf_qcd']
        + df['process_fraction_ttbar'] * df['ttbar_corrected_classic_ff']
    )
    """
    return df1, df2, clip_mask_tau1, clip_mask_tau2


@contextmanager
def temporary_extract_scaler(
    model: t.nn.Module,
    shift_attr: str = "_scaler_shift",
    scale_attr: str = "_scaler_scale",
) -> Iterable[Tuple[t.Tensor, t.Tensor]]:

    try:
        shift, scale = getattr(model, shift_attr), getattr(model, scale_attr)
    except AttributeError as e:
        msg = f"Model does not have attributes {shift_attr} and/or {scale_attr}"
        logger.error(msg)
        raise AttributeError(msg) from e

    _shift, _scale = shift.clone(), scale.clone()

    shift.fill_(0.0)
    scale.fill_(1.0)

    try:
        yield model, _shift, _scale
    finally:
        shift.copy_(_shift)
        scale.copy_(_scale)


# ------- masks ----------

def load_masks_config(path: str | Path = MASKS_CONFIG_PATH) -> dict[str, list[str]]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f'Masks config not found: {config_path}')

    with open(config_path, 'r') as handle:
        raw = yaml.safe_load(handle) or {}

    masks = raw.get('masks', raw)

    if not isinstance(masks, dict):
        raise ValueError(f'Invalid masks config format in {config_path}: expected a mapping at root or under "masks"')

    normalized: dict[str, list[str]] = {}
    for name, expressions in masks.items():        
        if isinstance(expressions, str):
            normalized[name] = [expressions]
            continue
        if isinstance(expressions, list) and all(isinstance(expr, str) for expr in expressions):
            normalized[name] = expressions
            continue
        raise ValueError(f'Invalid expression list for mask "{name}" in {config_path}')
    
    logger.info('Loaded %d masks from %s', len(normalized), config_path)
    return normalized


def _build_mask_from_config(df: pd.DataFrame, mask_name: str) -> pd.Series:
    global MASKS_CONFIG

    if not MASKS_CONFIG:
        MASKS_CONFIG = load_masks_config()

    expressions = MASKS_CONFIG.get(mask_name)

    if not expressions:
        raise KeyError(f'Mask "{mask_name}" not found in {MASKS_CONFIG_PATH}')

    combined_expression = ' & '.join(f'({expr})' for expr in expressions)
    mask = df.eval(combined_expression, engine='python')
    result = mask.fillna(False).astype(bool)
    return result


def _apply_config_mask(df: pd.DataFrame, mask_name: str) -> pd.DataFrame:
    return df[_build_mask_from_config(df, mask_name)].copy()

def mask_preselection_tight(df):
    return _apply_config_mask(df, 'mask_preselection_tight')


def mask_preselection_tight_binary_classifier(df):
    return _apply_config_mask(df, 'mask_preselection_tight_binary_classifier')


def mask_preselection_for_estimator(df):
    return mask_preselection_tight(df)

def mask_DR(df):
    return _apply_config_mask(df, 'mask_DR')

def AR_like_tau1(df):
    return _apply_config_mask(df, 'AR_like_tau1')

def AR_like_tau2(df):
    return _apply_config_mask(df, 'AR_like_tau2')

def SR_like(df):
    return _apply_config_mask(df, 'SR_like')

def AR_tau1(df):                 # without SS/OS conditions !!!!!!!!!!!!11
    return _apply_config_mask(df, 'AR_tau1')

def AR_tau2(df):                 # without SS/OS conditions !!!!!!!!!!!!11
    return _apply_config_mask(df, 'AR_tau2')

def SR(df):                 # without SS/OS conditions !!!!!!!!!!!!11
    return _apply_config_mask(df, 'SR')


# ----------- other utils -----------

def select_njets_category(df, category_name):
    if category_name == 'njets_0':
        return df[df.njets == 0].copy()
    if category_name == 'njets_1':
        return df[df.njets == 1].copy()
    if category_name == 'njets_geq_2':
        return df[df.njets >= 2].copy()
    if category_name == 'njets_inclusive':
        return df[df.njets >= 0].copy()
    raise ValueError(f'Unknown njets category: {category_name}')


def _build_main_bins_by_variable() -> dict[str, np.ndarray]:
    return {
    # Existing defaults (kept identical to plot_complete_variables == False setup)
    'pt_1': np.linspace(30, 150, 31),
    'pt_2': np.linspace(30, 150, 31),
    'm_vis': np.linspace(0, 300, 31),
    'deltaR_ditaupair': np.linspace(0, 5, 21),
    'pt_vis': np.linspace(0, 150, 31),
    'pt_tt': np.linspace(0, 200, 31),
    'm_fastmtt': np.linspace(0, 220, 31),
    'eta_1': np.linspace(-2.5, 2.5, 31),
    'eta_2': np.linspace(-2.5, 2.5, 31),
    'met': np.linspace(0, 125, 31),
    'mt_1': np.linspace(0, 200, 31),
    'mt_2': np.linspace(0, 200, 31),

    # Additional variables for plot_complete_variables == True
    'jpt_1': np.linspace(0, 150, 31),
    'jpt_2': np.linspace(0, 150, 31),
    'jeta_1': np.linspace(-5, 5, 31),
    'jeta_2': np.linspace(-5, 5, 31),
    'pt_fastmtt': np.linspace(0, 220, 31),
    'njets': np.linspace(-0.5, 8.5, 10),
    'mt_tot': np.linspace(0, 400, 41),
    'mjj': np.linspace(0, 600, 31),
    'pt_dijet': np.linspace(0, 400, 41),
    'pt_ttjj': np.linspace(0, 200, 41),
    'deltaEta_jj': np.linspace(-6, 6, 31),
    'deltaR_jj': np.linspace(-6, 6, 31),
    'deltaR_1j1': np.linspace(-6, 6, 31),
    'deltaR_1j2': np.linspace(-6, 6, 31),
    'deltaR_2j1': np.linspace(-6, 6, 31),
    'deltaR_2j2': np.linspace(-6, 6, 31),
    'deltaR_12j1': np.linspace(-6, 6, 31),
    'deltaR_12j2': np.linspace(-6, 6, 31),
    'deltaEta_1j1': np.linspace(-6, 6, 31),
    'deltaEta_1j2': np.linspace(-6, 6, 31),
    'deltaEta_2j1': np.linspace(-6, 6, 31),
    'deltaEta_2j2': np.linspace(-6, 6, 31),
    'deltaEta_12j1': np.linspace(-6, 6, 31),
    'deltaEta_12j2': np.linspace(-6, 6, 31),
    'nbtag': np.linspace(-0.5, 4.5, 6),
    'iso_1': np.linspace(0, 0.15, 31),
    'iso_2': np.linspace(0.6, 1.2, 31),
    'tau_decaymode_1': np.linspace(-0.5, 12.5, 14),
    'tau_decaymode_2': np.linspace(-0.5, 12.5, 14),
    'mass_1': np.linspace(0, 0.10, 31),
    'mass_2': np.linspace(0, 2.0, 31),
    'metphi': np.linspace(-3.5, 3.5, 31)
    }


def _build_sampling_bins_from_main(
    main_bins_map: dict[str, np.ndarray],
    max_scale: float = 2.0,
) -> dict[str, np.ndarray]:
    sampling_bins: dict[str, np.ndarray] = {}
    for var_name, main_bins in main_bins_map.items():
        main_bins = np.asarray(main_bins, dtype=np.float64)
        if main_bins.ndim != 1 or main_bins.size < 2:
            continue

        x_min = float(main_bins[0])
        x_max_main = float(main_bins[-1])
        x_max_sampling = max_scale * x_max_main

        # Keep monotonic edges in pathological cases.
        if x_max_sampling <= x_min:
            x_max_sampling = x_max_main
            if x_max_sampling <= x_min:
                x_max_sampling = x_min + 1.0

        n_bins = int(main_bins.size - 1)
        sampling_bins[var_name] = np.linspace(x_min, x_max_sampling, n_bins + 1)

    return sampling_bins


def plot_all_var(data):
    hep.style.use(hep.style.CMS)

    all_var = list(data_complete.columns)

    all_var_dir = Path("/work/tapp/TauFF/NF4FF/Normalizing_Flow_tt/plots/embedding") / "all_var"
    all_var_dir.mkdir(parents=True, exist_ok=True)

    for x in all_var:
        fig, ax = plt.subplots(1, 1, figsize=(13, 10.4))

        CMS_CHANNEL_TITLE(ax)
        CMS_LUMI_TITLE(ax)
        CMS_LABEL(ax)

        try:
            n = ax.hist(data[x], bins=50)
            logger.info(f"Plotting {x}")
        except:
            logger.info(f"Can't histogram {x}")
            continue

        ax.set_xlabel(x)
        ax.set_ylabel("Events")

        ax.set_ylim(top=1.2*np.max(n[0]))
        #ax.set_yscale('log')

        fig.savefig(all_var_dir / f"{x}.png")
        plt.close(fig)


def initialize_runtime_context() -> None:
    """Load args, models, data and plotting metadata into module-level runtime context."""
    global args, variables, dim, training_variables_tag, variables_with_njets, device
    global mode_dir, include_njets_feature, resolved_tag
    global MASKS_CONFIG
    global classifier_features_qcd
    global chk_pth_model_AR_like_tau1, chk_pth_model_AR_like_tau2, chk_pth_model_SR_like_tau1, chk_pth_model_SR_like_tau2
    global model_AR_like_tau1, model_AR_like_tau2, model_SR_like_tau1, model_SR_like_tau2
    global data_complete, list_variables, labels, labels_short, list_xlabels, list_bins
    global main_plot_bins_by_variable, sampling_plot_bins_by_variable, plot_root_dir



    # Step 1: parse runtime arguments and core variable list
    with open(cfg_path['variables'], 'r') as f:
        variables = yaml.safe_load(f)[args.var]

    cfg_set = load_config(cfg_path['config_settings'])

    dim = len(variables)
    training_variables_tag = build_training_variables_tag(variables)
    variables_with_njets = ['njets'] + variables
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    MASKS_CONFIG = load_masks_config(MASKS_CONFIG_PATH)

    

    # Step 2: resolve model mode and load model checkpoints
    _MODE_DIR = {
        'grouped_njets_split': 'split_njets_0_1_ge2',
        'single_nf':           'no_njets_split',
        'conditional_nf':      'conditional_njets_input',
    }
    mode_dir = _MODE_DIR[args.model_mode]
    include_njets_feature = args.model_mode in ('grouped_njets_split', 'conditional_nf')    

    resolved_tag = resolve_training_tag(variables, mode_dir)
    logger.info('Using training tag: %s (exact computed: %s)', resolved_tag, training_variables_tag)

    chk_pth_model_AR_like_tau1 = f'{cfg_path["NF_results"]}/{mode_dir}/training_{resolved_tag}/tau1/all/AR-like/latest'
    chk_pth_model_SR_like_tau1 = f'{cfg_path["NF_results"]}/{mode_dir}/training_{resolved_tag}/tau1/all/SR-like/latest'
    chk_pth_model_AR_like_tau2 = f'{cfg_path["NF_results"]}/{mode_dir}/training_{resolved_tag}/tau2/all/AR-like/latest'
    chk_pth_model_SR_like_tau2 = f'{cfg_path["NF_results"]}/{mode_dir}/training_{resolved_tag}/tau2/all/SR-like/latest'

    config_AR_like_tau1 = load_saved_model_config(chk_pth_model_AR_like_tau1, config_path)
    config_SR_like_tau1 = load_saved_model_config(chk_pth_model_SR_like_tau1, config_path)
    config_AR_like_tau2 = load_saved_model_config(chk_pth_model_AR_like_tau2, config_path)
    config_SR_like_tau2 = load_saved_model_config(chk_pth_model_SR_like_tau2, config_path)
    
    logger.info('Loading models from RNVP checkpoints:')
    if args.model_mode == 'grouped_njets_split':
        model_AR_like_tau1 = load_grouped_qcd_njets_router(
            checkpoint_dir=chk_pth_model_AR_like_tau1,
            config_path=config_path,
            variables=variables,
            device=device,
        )
        model_SR_like_tau1 = load_grouped_qcd_njets_router(
            checkpoint_dir=chk_pth_model_SR_like_tau1,
            config_path=config_path,
            variables=variables,
            device=device,
        )
        model_AR_like_tau2 = load_grouped_qcd_njets_router(
            checkpoint_dir=chk_pth_model_AR_like_tau2,
            config_path=config_path,
            variables=variables,
            device=device,
        )
        model_SR_like_tau2 = load_grouped_qcd_njets_router(
            checkpoint_dir=chk_pth_model_SR_like_tau2,
            config_path=config_path,
            variables=variables,
            device=device,
        )
    elif args.model_mode == 'conditional_nf':
        model_AR_like_tau1 = load_conditional_flow(dim=dim, cfg=config_AR_like_tau1, checkpoint_path=f'{chk_pth_model_AR_like_tau1}/model_checkpoint.pth', device=device)
        model_SR_like_tau1 = load_conditional_flow(dim=dim, cfg=config_SR_like_tau1, checkpoint_path=f'{chk_pth_model_SR_like_tau1}/model_checkpoint.pth', device=device)
        model_AR_like_tau2 = load_conditional_flow(dim=dim, cfg=config_AR_like_tau2, checkpoint_path=f'{chk_pth_model_AR_like_tau2}/model_checkpoint.pth', device=device)
        model_SR_like_tau2 = load_conditional_flow(dim=dim, cfg=config_SR_like_tau2, checkpoint_path=f'{chk_pth_model_SR_like_tau2}/model_checkpoint.pth', device=device)
    elif args.model_mode == 'single_nf':
        model_AR_like_tau1 = load_flow(dim=dim, cfg=config_AR_like_tau1, checkpoint_path=f'{chk_pth_model_AR_like_tau1}/model_checkpoint.pth', device=device)
        model_SR_like_tau1 = load_flow(dim=dim, cfg=config_SR_like_tau1, checkpoint_path=f'{chk_pth_model_SR_like_tau1}/model_checkpoint.pth', device=device)
        model_AR_like_tau2 = load_flow(dim=dim, cfg=config_AR_like_tau2, checkpoint_path=f'{chk_pth_model_AR_like_tau2}/model_checkpoint.pth', device=device)
        model_SR_like_tau2 = load_flow(dim=dim, cfg=config_SR_like_tau2, checkpoint_path=f'{chk_pth_model_SR_like_tau2}/model_checkpoint.pth', device=device)
    else:
        raise ValueError(f'Unknown model mode: {args.model_mode}. Supported: "grouped_njets_split", "conditional_nf", "single_nf".')

    
    # Step 3: load data and plotting labels/binning
    data_complete = pd.read_feather(f'{cfg_path["datasets"]}/{args.embedding}/combined_data_updated.feather')

    #plot_all_var(data_complete)
    #exit()
    
    if args.plot_complete_variables:
        list_variables = cfg_set['variables']
        list_variables.remove('q_1')
        list_variables.remove('q_2')
        list_variables.remove('event')
        list_variables.remove('metphi')
    else:
        list_variables = variables


    with open(cfg_path['labels'], 'r') as f:
        labels = yaml.safe_load(f)['tt']
    with open(cfg_path['labels_short'], 'r') as f:
        labels_short = yaml.safe_load(f)['tt']
    
    list_xlabels = [labels[k] for k in list_variables]    

    bins_by_variable = _build_main_bins_by_variable()
    list_bins = [np.asarray(bins_by_variable[var]) for var in list_variables]
    main_plot_bins_by_variable = {var: np.asarray(bins_by_variable[var]) for var in list_variables}
    sampling_plot_bins_by_variable = _build_sampling_bins_from_main(main_plot_bins_by_variable, max_scale=2.0)



    # Step 4: prepare output directories and optional training diagnostics
    plot_root_dir = Path(cfg_path['plots']) / args.embedding / mode_dir / f"training_{resolved_tag}"
    
    if not plot_root_dir.exists():
        plot_root_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Plot output root: {plot_root_dir}")

    training_log_specs = [
        ('Tau 1 AR-like', chk_pth_model_AR_like_tau1),
        ('Tau 1 SR-like', chk_pth_model_SR_like_tau1),
        ('Tau 2 AR-like', chk_pth_model_AR_like_tau2),
        ('Tau 2 SR-like', chk_pth_model_SR_like_tau2),
    ]
    
    if args.plot_training_diagnostics:
        plot_training_histories(training_log_specs, plot_root_dir / 'training_diagnostics')


def _sample_nf_features_for_region(model, n_samples: int, reference_df: pd.DataFrame, model_mode: str, device: torch.device) -> np.ndarray:
    if n_samples <= 0:
        return np.empty((0, len(variables)), dtype=np.float32)

    with torch.no_grad():
        if model_mode == 'single_nf':
            sampled = model.sample(n_samples)

        elif model_mode == 'conditional_nf':
            if len(reference_df) > 0 and 'njets' in reference_df.columns:
                reference_njets = reference_df['njets'].to_numpy(dtype=np.float32)
                cond_np = np.random.choice(reference_njets, size=n_samples, replace=True).astype(np.float32)
            else:
                cond_np = np.zeros(n_samples, dtype=np.float32)

            cond = torch.from_numpy(cond_np).to(device=device, dtype=torch.float32).unsqueeze(1)
            sampled = model.sample(cond)

        else:  # grouped_njets_split
            if len(reference_df) > 0 and 'njets' in reference_df.columns:
                njets_ref = reference_df['njets'].to_numpy(dtype=np.float32)
                group_ref = np.where(njets_ref == 0, 0, np.where(njets_ref == 1, 1, 2)).astype(np.int64)
                sampled_groups = np.random.choice(group_ref, size=n_samples, replace=True)
            else:
                sampled_groups = np.zeros(n_samples, dtype=np.int64)

            sampled_chunks = []
            for group_idx in (0, 1, 2):
                n_group = int(np.sum(sampled_groups == group_idx))
                if n_group == 0:
                    continue
                sampled_chunks.append(model.models[group_idx].sample(n_group))

            if len(sampled_chunks) == 0:
                sampled = torch.empty((0, len(variables)), device=device)
            else:
                sampled = torch.cat(sampled_chunks, dim=0)
                perm = torch.randperm(sampled.shape[0], device=sampled.device)
                sampled = sampled[perm]

    return sampled.detach().cpu().numpy().astype(np.float32)


def plot_nf_sampling_training_variables(category_name: str, njets_title: str, data_preselected: pd.DataFrame) -> None:
    sampling_plot_dir = plot_root_dir / 'nf_sampling_validation' / category_name
    sampling_plot_dir.mkdir(parents=True, exist_ok=True)

    tau1_ar_data = AR_like_tau1(data_preselected)
    tau1_ar_data = tau1_ar_data[(tau1_ar_data.process == 0) & (tau1_ar_data.SS == True)].copy()

    tau2_ar_data = AR_like_tau2(data_preselected)
    tau2_ar_data = tau2_ar_data[(tau2_ar_data.process == 0) & (tau2_ar_data.SS == True)].copy()

    sr_data = SR_like(data_preselected)
    sr_data = sr_data[(sr_data.process == 0) & (sr_data.SS == True)].copy()

    panel_specs = [
        ("Tau 1 AR-like", tau1_ar_data, model_AR_like_tau1, '#d62728', 'tau1_arlike'),
        ("Tau 2 AR-like", tau2_ar_data, model_AR_like_tau2, '#2ca02c', 'tau2_arlike'),
        ("Tau 1 SR-like", sr_data, model_SR_like_tau1, '#ff7f0e', 'tau1_srlike'),
        ("Tau 2 SR-like", sr_data, model_SR_like_tau2, '#1f77b4', 'tau2_srlike'),
    ]
    
    n_samples = 100000
    # ----- one plot vor every variable -----
    for var in variables:
        fig, axes = plt.subplots(2, 2, figsize=(19.2, 14.4), sharex=False, sharey=False)
        flat_axes = axes.flatten()
        fixed_bins = sampling_plot_bins_by_variable.get(var)

        # ----- in each plot: one plot of each region ------
        for axis, (title, data_df, model, color, tag) in zip(flat_axes, panel_specs):

            CMS_CHANNEL_TITLE(axis)
            CMS_LUMI_TITLE(axis)
            CMS_LABEL(axis)
            CMS_NJETS_TITLE(axis, title=njets_title)

            if data_df.empty:
                axis.text(0.5, 0.5, 'No data events', ha='center', va='center', transform=axis.transAxes)
                axis.set_title(title)
                axis.set_xlabel(labels.get(var, var))
                axis.set_ylabel('Density')
                axis.set_yscale('log')
                axis.grid(True, linestyle=':', alpha=0.35)
                if fixed_bins is not None:
                    axis.set_xlim(float(fixed_bins[0]), float(fixed_bins[-1]))
                continue

            sampled_np = _sample_nf_features_for_region(
                model=model,
                n_samples=n_samples,
                reference_df=data_df,
                model_mode=args.model_mode,
                device=device,
            )

            data_values = data_df[var].to_numpy(dtype=np.float32)
            sampled_values = sampled_np[:, variables.index(var)] if sampled_np.size else np.array([], dtype=np.float32)

            data_values = data_values[np.isfinite(data_values)]
            sampled_values = sampled_values[np.isfinite(sampled_values)]

            if data_values.size == 0 or sampled_values.size == 0:
                axis.text(0.5, 0.5, 'No finite values', ha='center', va='center', transform=axis.transAxes)
                axis.set_title(title)
                axis.set_xlabel(labels.get(var, var))
                axis.set_ylabel('Density')
                axis.set_yscale('log')
                axis.grid(True, linestyle=':', alpha=0.35)
                if fixed_bins is not None:
                    axis.set_xlim(float(fixed_bins[0]), float(fixed_bins[-1]))
                continue

            bins = fixed_bins
            if bins is None:
                combined = np.concatenate([data_values, sampled_values])
                if np.allclose(combined.min(), combined.max()):
                    half_width = max(abs(float(combined.min())) * 0.05, 1.0)
                    bins = np.linspace(float(combined.min()) - half_width, float(combined.max()) + half_width, 31)
                else:
                    bins = np.quantile(combined, np.linspace(0.0, 1.0, 41))
                    bins = np.unique(bins)
                    if bins.size < 10:
                        bins = np.linspace(float(combined.min()), float(combined.max()), 31)

            n_sample,_ ,_ = axis.hist(
                sampled_values,
                bins=bins,
                density=True,
                histtype='stepfilled',
                alpha=0.35,
                color=color,
                label=f'NF sampled ({n_samples})',
            )
            n_data,_ ,_ = axis.hist(
                data_values,
                bins=bins,
                density=True,
                histtype='step',
                linewidth=1.8,
                color='black',
                label=f'Data ({len(data_values)})',
            )
            
            axis.set_title(title, fontsize=25, pad=10)
            axis.set_xlabel(labels.get(var, var))
            axis.set_ylabel('Density')
            axis.set_yscale('log')
            axis.grid(True, linestyle=':', alpha=0.35)
            axis.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.0, 0.9))
            axis.set_xlim(float(bins[0]), float(bins[-1]))
            axis.set_ylim(top=5*np.max([np.max(n_sample), np.max(n_data)]))

        fig.suptitle(
            f'NF sampled vs data ({category_name}, {njets_title})\nTraining variable: {labels.get(var, var)}',
            fontsize=28,
            y=1.01,
        )
        fig.tight_layout()
        fig.savefig(sampling_plot_dir / f'nf_sample_vs_data_{var}.png', bbox_inches='tight')
        fig.savefig(sampling_plot_dir / f'nf_sample_vs_data_{var}.pdf', bbox_inches='tight')
        plt.close(fig)

    logger.info('Saved NF sampling validation plots to %s', sampling_plot_dir)
        
    for title, data_df, model, color, tag in panel_specs:
        sampled_np = _sample_nf_features_for_region(
                model=model,
                n_samples=n_samples,
                reference_df=data_df,
                model_mode=args.model_mode,
                device=device,
            )
        corr_matrix_nfsample_data(data_df, sampled_np, variables, title, tag, sampling_plot_dir)
    
    logger.info(f'Saved NF correlation matrix plots to {sampling_plot_dir}')



# Taylor coefficient analysis


def _compute_first_order_tcs(model, data_df: pd.DataFrame, n_events: int = 3_000) -> dict:
    """
    Return variance-normalized mean-absolute first-order Taylor coefficients of
    log p(x) w.r.t. each input feature.

    Normalization is done with per-feature standard deviation, i.e.
        c_i^(1,norm) = sigma_i * <| d log p / d x_i |>
    where sigma_i = sqrt(var(x_i)) measured on `data_df`.

    Handles all three model modes:
      - single_nf         : model(X), X shape (N, dim)
      - conditional_nf    : model(X), X shape (N, 1+dim), col-0 = njets
      - grouped_njets_split: per-sub-model TCs, weighted average over njets groups
    """

    if data_df.empty:
        return {}

    _reduce = lambda x: float(x.abs().mean().detach().cpu())

    feature_std = np.sqrt(
        np.nan_to_num(
            data_df[variables].to_numpy(dtype=np.float32).var(axis=0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    )

    if args.model_mode == 'grouped_njets_split':
        feat_names = list(variables)
        njets_arr = data_df['njets'].to_numpy(dtype=np.float32)
        group_labels = np.where(njets_arr == 0, 0, np.where(njets_arr == 1, 1, 2))
        tc_idx = [(i,) for i in range(len(variables))]
        tc_accum = np.zeros(len(variables))
        total_w = 0
        for grp_idx, sub_model in enumerate(model.models):
            grp_mask = group_labels == grp_idx
            n_grp = int(grp_mask.sum())
            if n_grp == 0:
                continue
            X_sub = data_df.iloc[np.where(grp_mask)[0]][variables].to_numpy(dtype=np.float32)
            if len(X_sub) > n_events:
                rng_idx = np.random.choice(len(X_sub), n_events, replace=False)
                X_sub = X_sub[rng_idx]
            X_t = torch.tensor(X_sub, dtype=torch.float32, device=device)

            # Pass raw data directly to the full model forward so that autograd
            # accounts for all internal transforms: cut-preprocessing (log),
            # tail-preprocessing (asinh/log1p), and z-score scaler.
            ext = _ta_extend(deepcopy(sub_model))
            with torch.enable_grad():
                tcs = ext.get_tc(
                    forward_kwargs_tctensor_key='X',
                    forward_kwargs={'X': X_t},
                    tc_idx_list=tc_idx,
                    reduce_func=_reduce,
                    eval_max_output_node_only=False,
                    selected_output_node=None,
                )
            raw_tc_grp = np.array([tcs.get((i,), 0.0) for i in range(len(variables))], dtype=np.float64)
            tc_accum += n_grp * raw_tc_grp
            total_w += n_grp
        if total_w == 0:
            return {}
        tc_accum /= total_w
        tc_accum = tc_accum * feature_std
        return {feat_names[i]: float(tc_accum[i]) for i in range(len(feat_names))}

    # --- single_nf or conditional_nf ---
    # For conditional_nf, njets is a discrete conditioning variable — not modeled by the flow.
    # We still pass it as col-0 for the forward pass, but only compute TCs for the feature columns.
    input_names = (['njets'] + list(variables)) if args.model_mode == 'conditional_nf' else list(variables)
    feat_names = list(variables)  # always only the flow features
    feat_offset = 1 if args.model_mode == 'conditional_nf' else 0
    X_np = data_df[input_names].to_numpy(dtype=np.float32)
    if len(X_np) > n_events:
        rng_idx = np.random.choice(len(X_np), n_events, replace=False)
        X_np = X_np[rng_idx]
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    # indices into the full input tensor (offset by 1 for conditional_nf to skip njets)
    tc_idx = [(feat_offset + i,) for i in range(len(feat_names))]
    # Pass raw data directly to the full model forward so that autograd
    # accounts for all internal transforms: cut-preprocessing (log),
    # tail-preprocessing (asinh/log1p), and z-score scaler.
    ext = _ta_extend(deepcopy(model))
    with torch.enable_grad():
        tcs = ext.get_tc(
            forward_kwargs_tctensor_key='X',
            forward_kwargs={'X': X_t},
            tc_idx_list=tc_idx,
            reduce_func=_reduce,
            eval_max_output_node_only=False,
            selected_output_node=None,
        )
    raw_tc = np.array([tcs.get((feat_offset + i,), 0.0) for i in range(len(feat_names))], dtype=np.float64)
    raw_tc = raw_tc * feature_std
    return {feat_names[i]: float(raw_tc[i]) for i in range(len(feat_names))}


def _compute_second_order_tcs(model, data_df: pd.DataFrame, n_events: int = 3_000):
    """
    Return variance-normalized mean-absolute second-order Taylor coefficients
    of log p(x) as a symmetric matrix of shape (n_feat, n_feat).
    Entry [i, j] corresponds to
        sigma_i * sigma_j * <|d² log p / dx_i dx_j|>
    where sigma_k = sqrt(var(x_k)) measured on `data_df`.

    Handles all three model modes the same way as _compute_first_order_tcs.
    Returns (matrix, feat_names) or None on failure.
    """
    from tayloranalysis import extend_model as _ta_extend

    if data_df.empty:
        return None

    _reduce = lambda x: float(x.abs().mean().detach().cpu())

    feature_std = np.sqrt(
        np.nan_to_num(
            data_df[variables].to_numpy(dtype=np.float32).var(axis=0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    )

    if args.model_mode == 'grouped_njets_split':
        feat_names = list(variables)
        n_feat = len(feat_names)
        njets_arr = data_df['njets'].to_numpy(dtype=np.float32)
        group_labels = np.where(njets_arr == 0, 0, np.where(njets_arr == 1, 1, 2))
        tc_idx = [(i, j) for i in range(n_feat) for j in range(n_feat)]
        tc_accum = np.zeros((n_feat, n_feat))
        total_w = 0
        for grp_idx, sub_model in enumerate(model.models):
            grp_mask = group_labels == grp_idx
            n_grp = int(grp_mask.sum())
            if n_grp == 0:
                continue
            X_sub = data_df.iloc[np.where(grp_mask)[0]][variables].to_numpy(dtype=np.float32)
            if len(X_sub) > n_events:
                rng_idx = np.random.choice(len(X_sub), n_events, replace=False)
                X_sub = X_sub[rng_idx]
            X_t = torch.tensor(X_sub, dtype=torch.float32, device=device)

            # Pass raw data directly to the full model forward so that autograd
            # accounts for all internal transforms: cut-preprocessing (log),
            # tail-preprocessing (asinh/log1p), and z-score scaler.
            ext = _ta_extend(deepcopy(sub_model))
            with torch.enable_grad():
                tcs = ext.get_tc(
                    forward_kwargs_tctensor_key='X',
                    forward_kwargs={'X': X_t},
                    tc_idx_list=tc_idx,
                    reduce_func=_reduce,
                    eval_max_output_node_only=False,
                    selected_output_node=None,
                )
            mat = np.array([[tcs.get((i, j), 0.0) for j in range(n_feat)] for i in range(n_feat)], dtype=np.float64)
            tc_accum += n_grp * mat
            total_w += n_grp
        if total_w == 0:
            return None
        mat_norm = (tc_accum / total_w) * np.outer(feature_std, feature_std)
        return mat_norm, feat_names

    # --- single_nf or conditional_nf ---
    # njets is discrete and conditions the flow only — exclude it from TC indices.
    input_names = (['njets'] + list(variables)) if args.model_mode == 'conditional_nf' else list(variables)
    feat_names = list(variables)  # always only the flow features
    feat_offset = 1 if args.model_mode == 'conditional_nf' else 0
    n_feat = len(feat_names)
    X_np = data_df[input_names].to_numpy(dtype=np.float32)
    if len(X_np) > n_events:
        rng_idx = np.random.choice(len(X_np), n_events, replace=False)
        X_np = X_np[rng_idx]
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    tc_idx = [(feat_offset + i, feat_offset + j) for i in range(n_feat) for j in range(n_feat)]
    # Pass raw data directly to the full model forward so that autograd
    # accounts for all internal transforms: cut-preprocessing (log),
    # tail-preprocessing (asinh/log1p), and z-score scaler.
    ext = _ta_extend(deepcopy(model))
    with torch.enable_grad():
        tcs = ext.get_tc(
            forward_kwargs_tctensor_key='X',
            forward_kwargs={'X': X_t},
            tc_idx_list=tc_idx,
            reduce_func=_reduce,
            eval_max_output_node_only=False,
            selected_output_node=None,
        )
    mat = np.array([[tcs.get((feat_offset + i, feat_offset + j), 0.0) for j in range(n_feat)] for i in range(n_feat)], dtype=np.float64)
    mat = mat * np.outer(feature_std, feature_std)
    return mat, feat_names


def _compute_first_order_output_side_tcs(model, n_events: int = 3000) -> dict:
    """
    Compute first-order Taylor coefficients of the inverse flow x = f^{-1}(z)
    w.r.t. each latent variable z_k.

    Returns:
        { 'z0': tc0, 'z1': tc1, ... }
    """

    if args.model_mode == 'grouped_njets_split':
        logger.warning('First-order output-side TC computation is not implemented for grouped_njets_split mode.')
        return {}

    # 1. Sample latent variables
    dim = model.dim  # latent dimension = data dimension
    z_np = np.random.randn(n_events, dim).astype(np.float32)
    z_t = torch.tensor(z_np, dtype=torch.float32, device=device)

    # 2. Prepare extended model for TC computation
    ext = _ta_extend(deepcopy(model))

    # 3. Indices of latent variables
    tc_idx = [(i,) for i in range(dim)]

    # 4. Compute TCs of x = f^{-1}(z) w.r.t. z
    with torch.enable_grad():
        tcs = ext.get_tc(
            forward_kwargs_tctensor_key='Z',
            forward_kwargs={'Z': z_t},
            tc_idx_list=tc_idx,
            reduce_func=lambda x: float(x.abs().mean().detach().cpu()),
            eval_max_output_node_only=False,
            selected_output_node=None,
            inverse=True,  # <-- IMPORTANT: use inverse flow
        )

    # 5. Extract and return
    raw_tc = np.array([tcs.get((i,), 0.0) for i in range(dim)], dtype=np.float64)

    return {f'z{i}': float(raw_tc[i]) for i in range(dim)}


# Plotting




def plot_nf_taylor_analysis(output_dir: Path) -> None:
    """
    Compute and plot first-order Taylor coefficients for all four NF models.
    Produces a 2x2 figure with horizontal bar charts sorted by |TC| magnitude.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data_pre = mask_preselection_for_estimator(data_complete)
    tau1_ar = AR_like_tau1(data_pre)
    tau2_ar = AR_like_tau2(data_pre)
    tau1_ar = tau1_ar[(tau1_ar.process == 0) & (tau1_ar.SS == True)].copy()
    tau2_ar = tau2_ar[(tau2_ar.process == 0) & (tau2_ar.SS == True)].copy()

    tau1_sr = SR_like(data_pre)
    tau2_sr = SR_like(data_pre)
    tau1_sr = tau1_sr[(tau1_sr.process == 0) & (tau1_sr.SS == True)].copy()    
    tau2_sr = tau2_sr[(tau2_sr.process == 0) & (tau2_sr.SS == True)].copy()

    panel_specs = [
        ('Tau1 AR-like',   tau1_ar,   model_AR_like_tau1,   '#d62728'),
        ('Tau2 AR-like',   tau2_ar,   model_AR_like_tau2,   '#2ca02c'),
        ('Tau1 SR-like',   tau1_sr,   model_SR_like_tau1,   '#ff7f0e'),
        ('Tau2 SR-like',   tau2_sr,   model_SR_like_tau2,   '#1f77b4'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    flat_axes = axes.flatten()

    for axis, (title, data_df, model, color) in zip(flat_axes, panel_specs):
        logger.info('Computing first-order Taylor coefficients for %s ...', title)
        try:
            tc_dict = _compute_first_order_tcs(model, data_df)
        except Exception as exc:
            logger.warning('Taylor analysis failed for %s: %s', title, exc)
            axis.text(0.5, 0.5, f'Failed:\n{exc}', ha='center', va='center',
                      transform=axis.transAxes, fontsize=8)
            axis.set_title(title)
            add_cms_privatework_lumi_row(axis)

            single_fig, single_axis = plt.subplots(figsize=(9.5, 7.2))
            single_axis.text(0.5, 0.5, f'Failed:\n{exc}', ha='center', va='center',
                             transform=single_axis.transAxes, fontsize=10)
            single_axis.set_title(title)
            add_cms_privatework_lumi_row(single_axis, fontsize=10)
            single_fig.suptitle(
                f'First-order Taylor coefficients  —  {title}  —  {args.model_mode}\n'
                r'$\tilde{c_i} = \sigma_i\,\langle\,|\,\partial \log p\,/\,\partial x_i\,|\,\rangle$',
                fontsize=14,
                y=0.955,
            )
            single_fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
            title_slug = _slugify_plot_label(title)
            single_fig.savefig(output_dir / f'taylor_coefficients_1st_order_{title_slug}.png', bbox_inches='tight')
            single_fig.savefig(output_dir / f'taylor_coefficients_1st_order_{title_slug}.pdf', bbox_inches='tight')
            plt.close(single_fig)
            continue

        if not tc_dict:
            axis.text(0.5, 0.5, 'No data', ha='center', va='center', transform=axis.transAxes)
            axis.set_title(title)
            add_cms_privatework_lumi_row(axis)

            single_fig, single_axis = plt.subplots(figsize=(9.5, 7.2))
            single_axis.text(0.5, 0.5, 'No data', ha='center', va='center', transform=single_axis.transAxes)
            single_axis.set_title(title)
            add_cms_privatework_lumi_row(single_axis, fontsize=10)
            single_fig.suptitle(
                f'First-order Taylor coefficients  —  {title}  —  {args.model_mode}\n'
                r'$\tilde{c_i} = \sigma_i\,\langle\,|\,\partial \log p\,/\,\partial x_i\,|\,\rangle$',
                fontsize=14,
                y=0.955,
            )
            single_fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
            title_slug = _slugify_plot_label(title)
            single_fig.savefig(output_dir / f'taylor_coefficients_1st_order_{title_slug}.png', bbox_inches='tight')
            single_fig.savefig(output_dir / f'taylor_coefficients_1st_order_{title_slug}.pdf', bbox_inches='tight')
            plt.close(single_fig)
            continue

        # sort ascending so the largest bar appears at the top of the chart
        sorted_items = sorted(tc_dict.items(), key=lambda kv: abs(kv[1]))
        display_names = [labels_short.get(k, k) for k, _ in sorted_items]
        tc_values = [float(v) for _, v in sorted_items]

        y_pos = np.arange(len(display_names))
        axis.barh(y_pos, tc_values, color=color, alpha=0.75, edgecolor='none')
        axis.set_yticks(y_pos)
        axis.set_yticklabels(display_names, fontsize=9)
        axis.set_xlabel(r'$\sigma_i\,\langle\,|\,\partial_i \log p(x)\,|\,\rangle$')
        #axis.set_title(title)
        add_cms_privatework_lumi_row(axis)
        axis.grid(True, axis='x', linestyle=':', alpha=0.4)
        axis.tick_params(direction='in')

        single_fig, single_axis = plt.subplots(figsize=(6, 5))
        single_axis.barh(y_pos, tc_values, color=color, alpha=0.75, edgecolor='none')
        single_axis.set_yticks(y_pos)
        single_axis.set_yticklabels(display_names, fontsize=10)
        single_axis.set_xlabel(r"$\tilde{c_i}$")
        #single_axis.set_xlabel(r'$\sigma_i\,\langle\,|\,\partial_i \log p(x)\,|\,\rangle$')
        #single_axis.set_title(title)
        add_cms_privatework_lumi_row(single_axis, fontsize=10)
        single_axis.grid(True, axis='x', linestyle=':', alpha=0.4)
        single_axis.tick_params(direction='in')
        single_fig.suptitle(
            f'First-order Taylor coefficients  — {title}  — {args.model_mode}\n'
            r'$\tilde{c_i} = \sigma_i\,\langle\,|\,\partial \log p\,/\,\partial x_i\,|\,\rangle$',
            fontsize=14,
            y=0.955,
        )
        single_fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
        title_slug = _slugify_plot_label(title)
        single_fig.savefig(output_dir / f'taylor_coefficients_1st_order_{title_slug}.png', bbox_inches='tight')
        single_fig.savefig(output_dir / f'taylor_coefficients_1st_order_{title_slug}.pdf', bbox_inches='tight')
        plt.close(single_fig)

    fig.suptitle(
        f'First-order Taylor coefficients  — {args.model_mode}\n'
        r'$\tilde{c_i} = \sigma_i\,\langle\,|\,\partial \log p\,/\,\partial x_i\,|\,\rangle$',
        fontsize=15, y=0.955,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    fig.savefig(output_dir / 'taylor_coefficients_1st_order.png', bbox_inches='tight')
    fig.savefig(output_dir / 'taylor_coefficients_1st_order.pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved first-order Taylor coefficient plots (combined + individual) to %s', output_dir)


def plot_nf_taylor_analysis_output(output_dir: Path) -> None:
    """
    Compute and plot first-order *output-side* Taylor coefficients for all four NF models.
    Produces a 2x2 figure with horizontal bar charts sorted by |TC| magnitude.

    Output-side TCs measure sensitivity of the inverse flow x = f^{-1}(z)
    w.r.t. each latent variable z_k.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data_pre = mask_preselection_for_estimator(data_complete)
    tau1_ar = AR_like_tau1(data_pre)
    tau2_ar = AR_like_tau2(data_pre)
    tau1_ar = tau1_ar[(tau1_ar.process == 0) & (tau1_ar.SS == True)].copy()
    tau2_ar = tau2_ar[(tau2_ar.process == 0) & (tau2_ar.SS == True)].copy()

    tau1_sr = SR_like(data_pre)
    tau2_sr = SR_like(data_pre)
    tau1_sr = tau1_sr[(tau1_sr.process == 0) & (tau1_sr.SS == True)].copy()    
    tau2_sr = tau2_sr[(tau2_sr.process == 0) & (tau2_sr.SS == True)].copy()

    panel_specs = [
        ('Tau1 AR-like',   tau1_ar,   model_AR_like_tau1,   '#d62728'),
        ('Tau2 AR-like',   tau2_ar,   model_AR_like_tau2,   '#2ca02c'),
        ('Tau1 SR-like',   tau1_sr,   model_SR_like_tau1,   '#ff7f0e'),
        ('Tau2 SR-like',   tau2_sr,   model_SR_like_tau2,   '#1f77b4'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    flat_axes = axes.flatten()

    for axis, (title, data_df, model, color) in zip(flat_axes, panel_specs):
        logger.info('Computing OUTPUT-side first-order Taylor coefficients for %s ...', title)

        try:
            tc_dict = _compute_first_order_output_side_tcs(model)
        except Exception as exc:
            logger.warning('Output-side Taylor analysis failed for %s: %s', title, exc)
            axis.text(0.5, 0.5, f'Failed:\n{exc}', ha='center', va='center',
                      transform=axis.transAxes, fontsize=8)
            axis.set_title(title)
            add_cms_privatework_lumi_row(axis)

            single_fig, single_axis = plt.subplots(figsize=(9.5, 7.2))
            single_axis.text(0.5, 0.5, f'Failed:\n{exc}', ha='center', va='center',
                             transform=single_axis.transAxes, fontsize=10)
            single_axis.set_title(title)
            add_cms_privatework_lumi_row(single_axis, fontsize=10)
            single_fig.suptitle(
                f'Output-side first-order Taylor coefficients  —  {title}  —  {args.model_mode}\n'
                r'$\langle\,|\,\partial x / \partial z_k\,|\,\rangle$',
                fontsize=14,
                y=0.955,
            )
            single_fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
            title_slug = _slugify_plot_label(title)
            single_fig.savefig(output_dir / f'taylor_coefficients_output_1st_order_{title_slug}.png', bbox_inches='tight')
            single_fig.savefig(output_dir / f'taylor_coefficients_output_1st_order_{title_slug}.pdf', bbox_inches='tight')
            plt.close(single_fig)
            continue

        if not tc_dict:
            axis.text(0.5, 0.5, 'No data', ha='center', va='center', transform=axis.transAxes)
            axis.set_title(title)
            add_cms_privatework_lumi_row(axis)

            single_fig, single_axis = plt.subplots(figsize=(9.5, 7.2))
            single_axis.text(0.5, 0.5, 'No data', ha='center', va='center', transform=single_axis.transAxes)
            single_axis.set_title(title)
            add_cms_privatework_lumi_row(single_axis, fontsize=10)
            single_fig.suptitle(
                f'Output-side first-order Taylor coefficients  —  {title}  —  {args.model_mode}\n'
                r'$\langle\,|\,\partial x / \partial z_k\,|\,\rangle$',
                fontsize=14,
                y=0.955,
            )
            single_fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
            title_slug = _slugify_plot_label(title)
            single_fig.savefig(output_dir / f'taylor_coefficients_output_1st_order_{title_slug}.png', bbox_inches='tight')
            single_fig.savefig(output_dir / f'taylor_coefficients_output_1st_order_{title_slug}.pdf', bbox_inches='tight')
            plt.close(single_fig)
            continue

        # sort ascending so the largest bar appears at the top
        sorted_items = sorted(tc_dict.items(), key=lambda kv: abs(kv[1]))
        display_names = [k for k, _ in sorted_items]  # latent dims: z0, z1, ...
        tc_values = [float(v) for _, v in sorted_items]

        y_pos = np.arange(len(display_names))
        axis.barh(y_pos, tc_values, color=color, alpha=0.75, edgecolor='none')
        axis.set_yticks(y_pos)
        axis.set_yticklabels(display_names, fontsize=9)
        axis.set_xlabel(r'$\langle\,|\,\partial x / \partial z_k\,|\,\rangle$')
        add_cms_privatework_lumi_row(axis)
        axis.grid(True, axis='x', linestyle=':', alpha=0.4)
        axis.tick_params(direction='in')

        # individual figure
        single_fig, single_axis = plt.subplots(figsize=(6, 5))
        single_axis.barh(y_pos, tc_values, color=color, alpha=0.75, edgecolor='none')
        single_axis.set_yticks(y_pos)
        single_axis.set_yticklabels(display_names, fontsize=10)
        single_axis.set_xlabel(r"$\langle\,|\,\partial x / \partial z_k\,|\,\rangle$")
        add_cms_privatework_lumi_row(single_axis, fontsize=10)
        single_axis.grid(True, axis='x', linestyle=':', alpha=0.4)
        single_axis.tick_params(direction='in')
        single_fig.suptitle(
            f'Output-side first-order Taylor coefficients  — {title}  — {args.model_mode}\n'
            r'$\langle\,|\,\partial x / \partial z_k\,|\,\rangle$',
            fontsize=14,
            y=0.955,
        )
        single_fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
        title_slug = _slugify_plot_label(title)
        single_fig.savefig(output_dir / f'taylor_coefficients_output_1st_order_{title_slug}.png', bbox_inches='tight')
        single_fig.savefig(output_dir / f'taylor_coefficients_output_1st_order_{title_slug}.pdf', bbox_inches='tight')
        plt.close(single_fig)

    fig.suptitle(
        f'Output-side first-order Taylor coefficients  — {args.model_mode}\n'
        r'$\langle\,|\,\partial x / \partial z_k\,|\,\rangle$',
        fontsize=15, y=0.955,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    fig.savefig(output_dir / 'taylor_coefficients_output_1st_order.png', bbox_inches='tight')
    fig.savefig(output_dir / 'taylor_coefficients_output_1st_order.pdf', bbox_inches='tight')
    plt.close(fig)

    logger.info('Saved OUTPUT-side first-order Taylor coefficient plots (combined + individual) to %s', output_dir)



def plot_nf_second_order_covariance(output_dir: Path) -> None:
    """
    Compute and plot second-order Taylor coefficient matrices for all four NF models.
    Each model produces a heatmap of mean |d² log p / dx_i dx_j|.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data_pre = mask_preselection_for_estimator(data_complete)
    tau1_ar = AR_like_tau1(data_pre)
    tau2_ar = AR_like_tau2(data_pre)
    tau1_ar = tau1_ar[(tau1_ar.process == 0) & (tau1_ar.SS == True)].copy()
    tau2_ar = tau2_ar[(tau2_ar.process == 0) & (tau2_ar.SS == True)].copy()

    tau1_sr = SR_like(data_pre)
    tau2_sr = SR_like(data_pre)
    tau1_sr = tau1_sr[(tau1_sr.process == 0) & (tau1_sr.SS == True)].copy()    
    tau2_sr = tau2_sr[(tau2_sr.process == 0) & (tau2_sr.SS == True)].copy()

    panel_specs = [
        ('Tau1 AR-like',   tau1_ar,   model_AR_like_tau1),
        ('Tau2 AR-like',   tau2_ar,   model_AR_like_tau2),
        ('Tau1 SR-like',   tau1_sr,   model_SR_like_tau1),
        ('Tau2 SR-like',   tau2_sr,   model_SR_like_tau2),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    flat_axes = axes.flatten()

    for axis, (title, data_df, model) in zip(flat_axes, panel_specs):
        logger.info('Computing second-order Taylor coefficients for %s ...', title)
        try:
            result = _compute_second_order_tcs(model, data_df)
        except Exception as exc:
            logger.warning('Second-order Taylor analysis failed for %s: %s', title, exc)
            axis.text(0.5, 0.5, f'Failed:\n{exc}', ha='center', va='center',
                      transform=axis.transAxes, fontsize=8)
            axis.set_title(title)
            add_cms_privatework_lumi_row(axis)

            single_fig, single_axis = plt.subplots(figsize=(9.5, 7.8))
            single_axis.text(0.5, 0.5, f'Failed:\n{exc}', ha='center', va='center',
                             transform=single_axis.transAxes, fontsize=10)
            single_axis.set_title(title)
            add_cms_privatework_lumi_row(single_axis, fontsize=10)
            single_fig.suptitle(
                f'Second-order Taylor coefficients  —  {title}  —  {args.model_mode}\n'
                r'$\tilde{c_{ij}} = \sigma_i\sigma_j\,\langle\,|\,\partial^2 \log p\,/\,\partial x_i\,\partial x_j\,|\,\rangle$',
                fontsize=14,
                y=0.955,
            )
            single_fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
            title_slug = _slugify_plot_label(title)
            single_fig.savefig(output_dir / f'taylor_coefficients_2nd_order_{title_slug}.png', bbox_inches='tight')
            single_fig.savefig(output_dir / f'taylor_coefficients_2nd_order_{title_slug}.pdf', bbox_inches='tight')
            plt.close(single_fig)
            continue

        if result is None:
            axis.text(0.5, 0.5, 'No data', ha='center', va='center', transform=axis.transAxes)
            axis.set_title(title)
            add_cms_privatework_lumi_row(axis)

            single_fig, single_axis = plt.subplots(figsize=(9.5, 7.8))
            single_axis.text(0.5, 0.5, 'No data', ha='center', va='center', transform=single_axis.transAxes)
            single_axis.set_title(title)
            add_cms_privatework_lumi_row(single_axis, fontsize=10)
            single_fig.suptitle(
                f'Second-order Taylor coefficients  —  {title}  —  {args.model_mode}\n'
                r'$\tilde{c_{ij}} = \sigma_i\sigma_j\,\langle\,|\,\partial^2 \log p\,/\,\partial x_i\,\partial x_j\,|\,\rangle$',
                fontsize=14,
                y=0.955,
            )
            single_fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
            title_slug = _slugify_plot_label(title)
            single_fig.savefig(output_dir / f'taylor_coefficients_2nd_order_{title_slug}.png', bbox_inches='tight')
            single_fig.savefig(output_dir / f'taylor_coefficients_2nd_order_{title_slug}.pdf', bbox_inches='tight')
            plt.close(single_fig)
            continue

        mat, feat_names = result
        display_names = [labels_short.get(k, k) for k in feat_names]
        n_feat = len(display_names)

        im = axis.imshow(mat, aspect='auto', cmap='viridis')
        axis.set_xticks(np.arange(n_feat))
        axis.set_yticks(np.arange(n_feat))
        axis.set_xticklabels(display_names, rotation=45, ha='right', fontsize=9)
        axis.set_yticklabels(display_names, fontsize=9)
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

        vmax = mat.max() if mat.max() > 0 else 1.0
        for i in range(n_feat):
            for j in range(n_feat):
                text_color = 'white' if mat[i, j] < 0.6 * vmax else 'black'
                axis.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                          fontsize=7, color=text_color)

        axis.set_title(title)
        add_cms_privatework_lumi_row(axis)

        single_fig, single_axis = plt.subplots(figsize=(9.5, 7.8))
        im_single = single_axis.imshow(mat, aspect='auto', cmap='viridis')
        single_axis.set_xticks(np.arange(n_feat))
        single_axis.set_yticks(np.arange(n_feat))
        single_axis.set_xticklabels(display_names, rotation=45, ha='right', fontsize=9)
        single_axis.set_yticklabels(display_names, fontsize=9)
        single_fig.colorbar(im_single, ax=single_axis, fraction=0.046, pad=0.04)
        for i in range(n_feat):
            for j in range(n_feat):
                text_color = 'white' if mat[i, j] < 0.6 * vmax else 'black'
                single_axis.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=7, color=text_color)
        single_axis.set_title(title)
        add_cms_privatework_lumi_row(single_axis, fontsize=10)
        single_fig.suptitle(
            f'Second-order Taylor coefficients  — {title}  — {args.model_mode}\n'
            r'$\tilde{c_{ij}} = \sigma_i\sigma_j\,\langle\,|\,\partial^2 \log p\,/\,\partial x_i\,\partial x_j\,|\,\rangle$',
            fontsize=14,
            y=0.955,
        )
        single_fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
        title_slug = _slugify_plot_label(title)
        single_fig.savefig(output_dir / f'taylor_coefficients_2nd_order_{title_slug}.png', bbox_inches='tight')
        single_fig.savefig(output_dir / f'taylor_coefficients_2nd_order_{title_slug}.pdf', bbox_inches='tight')
        plt.close(single_fig)

    fig.suptitle(
        f'Second-order Taylor coefficients  — {args.model_mode}\n'
        r'$\tilde{c_{ij}} = \sigma_i\sigma_j\,\langle\,|\,\partial^2 \log p\,/\,\partial x_i\,\partial x_j\,|\,\rangle$',
        fontsize=15, y=0.955,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    fig.savefig(output_dir / 'taylor_coefficients_2nd_order.png', bbox_inches='tight')
    fig.savefig(output_dir / 'taylor_coefficients_2nd_order.pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved second-order Taylor coefficient plots (combined + individual) to %s', output_dir)


def plot_ar_data_with_clipping_info(
    var: str,
    bins: np.ndarray,
    xlabel: str,
    data_ar_os_full: pd.DataFrame,
    clipping_mask: np.ndarray,
    njets_title: str,
    output_dir: Path,
    tau_label: str,
) -> None:
    """
    Plot AR data showing both kept and excluded events (by clipping mask).
    
    Args:
        var: Variable name
        bins: Bin edges
        xlabel: X-axis label
        data_ar_os_full: Full AR data before clipping
        clipping_mask: Boolean mask indicating which events are kept (True) vs excluded (False)
        njets_title: Title for njets
        output_dir: Output directory for saving plots
    """
    # Separate kept and excluded events
    data_kept = data_ar_os_full[clipping_mask]
    data_excluded = data_ar_os_full[~clipping_mask]
    
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05},
        constrained_layout=True,
    )
    ax_main, ax_ratio = ax

    CMS_CHANNEL_TITLE([ax_main])
    CMS_LUMI_TITLE([ax_main])
    CMS_LABEL([ax_main])
    CMS_NJETS_TITLE([ax_main], title=njets_title)
    
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_widths = np.diff(bins)

    # Plot histograms
    counts_kept, _ = np.histogram(data_kept[var], bins=bins)
    counts_excluded, _ = np.histogram(data_excluded[var], bins=bins)

    counts_complete = counts_excluded + counts_kept
    excluded_fraction = np.divide(
        counts_excluded,
        counts_complete,
        out=np.zeros_like(counts_excluded, dtype=float),
        where=counts_complete != 0,
    )

    excluded_percent = 100.0 * (1.0 - float(np.mean(clipping_mask)))
    
    # Plot excluded events first (lighter)
    ax_main.bar(
        bin_centers,
        counts_excluded,
        width=bin_widths * 0.95,
        label=f'Excluded by clipping ({excluded_percent:.2f}%)',
        color='#ff7f0e',
        alpha=0.5,
        edgecolor='black',
        linewidth=0.7,
    )
    
    # Plot kept events on top
    ax_main.bar(
        bin_centers,
        counts_kept,
        bottom=counts_excluded,
        width=bin_widths * 0.95,
        label='Data complete',
        color='#1f77b4',
        alpha=0.5,
        edgecolor='black',
        linewidth=0.7,
    )

    ax_main.set_ylabel("Events", fontsize=20)
    ax_main.legend(loc='upper right', frameon=False, fontsize=18)
    ax_main.tick_params(direction='in', top=True, right=True)
    ax_main.set_ylim(top = 1.3 * max(counts_complete))

    ax_ratio.stairs(excluded_fraction, bins, color='black', linewidth=1.4)
    ax_ratio.axhline(0.0, color='gray', linestyle=':', linewidth=1.0)
    ax_ratio.set_ylim(0.0, 1.0)
    ax_ratio.set_ylabel("Excluded / Total", fontsize=15, loc='center')
    ax_ratio.set_xlabel(xlabel, fontsize=20)
    ax_ratio.grid(True, linestyle=':', alpha=0.6)
    ax_ratio.tick_params(direction='in', top=True, right=True)

    final_dir = output_dir / 'clipping_info'
    if not final_dir.exists():
        os.makedirs(final_dir)
    fig.savefig(final_dir / f'{var}_ar_clipping_{tau_label}.png', dpi=150)
    fig.savefig(final_dir / f'{var}_ar_clipping_{tau_label}.pdf')
    plt.close(fig)

def plot_ff_values(ff_tau1, ff_tau2, plot_dir, njets_title):

        logger.info('Plotting FF values for Tau 1 and Tau 2 ...')

        panel_specs = [
            ("Tau 1", ff_tau1, '#d62728'),
            ("Tau 2", ff_tau2, '#2ca02c'),
        ]

        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=False, sharey=False)
        flat_axes = axes.flatten()

        # ----- in each plot: one plot of each region ------
        for axis, (title, ff, color) in zip(flat_axes, panel_specs):

            CMS_CHANNEL_TITLE(axis)
            CMS_LUMI_TITLE(axis)
            CMS_LABEL(axis)
            CMS_NJETS_TITLE(axis, title=njets_title)


            ff_values = ff.to_numpy(dtype=np.float32)
            ff_values = ff_values[np.isfinite(ff_values)]

            if ff_values.size == 0:
                axis.text(0.5, 0.5, 'No finite values', ha='center', va='center', transform=axis.transAxes)
                axis.set_title(title)
                axis.set_ylabel('counts')
                axis.set_xlabel(r'$F_F$')
                axis.grid(True, linestyle=':', alpha=0.35)
                continue
            

            n_ff,_ ,_ = axis.hist(
                ff_values,
                bins=20,
                histtype='stepfilled',
                alpha=0.35,
                color=color,
                label=f'FF values',
            )
            
            axis.set_title(title)#, fontsize=25, pad=10)
            axis.set_ylabel('counts')
            axis.set_xlabel(r'$F_F$')
            axis.grid(True, linestyle=':', alpha=0.35)
            axis.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.0, 0.9))
            #axis.set_xlim(float(bins[0]), float(bins[-1]))
            axis.set_ylim(top=1.3*np.max(np.max(n_ff)))

        fig.tight_layout()
        logger.info('Saving FF value distribution plots to %s', plot_dir)
        fig.savefig(plot_dir / f'ff_values.png', bbox_inches='tight')
        fig.savefig(plot_dir / f'ff_values.pdf', bbox_inches='tight')
        plt.close(fig)


def run_plots_for_njets_category(category_name, njets_title):
    hep.style.use(hep.style.CMS)  # Use CMS style for all plots in this category

    category_plot_dir = plot_root_dir / category_name
    category_plot_dir.mkdir(parents=True, exist_ok=True)

    data_complete_njets = select_njets_category(data_complete, category_name)
    data_preselected = mask_preselection_for_estimator(data_complete_njets)
    
    # ----- plot NF sampling -----
    if args.plot_nf_sampling:
        plot_nf_sampling_training_variables(category_name, njets_title, data_preselected)
    if not args.plot_ff_results and not args.plot_ar_data_with_clipping and not args.plot_ff_values:
        return
    
    
    # ----- prepare data for FF estimation -----
    logger.info(
        f"Starting {category_name}: {len(data_complete_njets)} input events, {len(data_preselected)} after preselection"
    )

    # ----- Anti-DR -----

    data_AR_tau1 = AR_tau1(data_preselected)
    data_AR_tau1 = data_AR_tau1[data_AR_tau1.OS == True]
    data_AR_OS_tau1 = data_AR_tau1[(data_AR_tau1.process == 0)].copy()

    data_AR_tau2 = AR_tau2(data_preselected)
    data_AR_tau2 = data_AR_tau2[data_AR_tau2.OS == True]
    data_AR_OS_tau2 = data_AR_tau2[(data_AR_tau2.process == 0)].copy()

    data_SR = SR(data_preselected)
    data_SR_OS = data_SR[(data_SR.OS == True)]


    # ----- DR -----

    data_AR_like_tau1 = AR_like_tau1(data_preselected)
    data_AR_like_tau2 = AR_like_tau2(data_preselected)
    data_SR_like = SR_like(data_preselected)

    data_AR_like_SS_tau1 = data_AR_like_tau1[(data_AR_like_tau1.process == 0) & (data_AR_like_tau1.SS == True)]
    data_SR_like_SS_tau1 = data_SR_like[(data_SR_like.process == 0) & (data_SR_like.SS == True)]
    data_AR_like_SS_tau2 = data_AR_like_tau2[(data_AR_like_tau2.process == 0) & (data_AR_like_tau2.SS == True)]
    data_SR_like_SS_tau2 = data_SR_like[(data_SR_like.process == 0) & (data_SR_like.SS == True)]

    required_samples = {
        'AR_like_SS_tau1': data_AR_like_SS_tau1,
        'SR_like_SS_tau1': data_SR_like_SS_tau1,
        'AR_like_SS_tau2': data_AR_like_SS_tau2,
        'SR_like_SS_tau2': data_SR_like_SS_tau2,
    }

    empty_required = [name for name, sample in required_samples.items() if sample.empty]
    if empty_required:
        logger.warning('Skipping %s because required samples are empty: %s', category_name, ', '.join(empty_required))
        return

    

    global_ff_tau1 = len(data_SR_like_SS_tau1) / len(data_AR_like_SS_tau1)
    global_ff_tau2 = len(data_SR_like_SS_tau2) / len(data_AR_like_SS_tau2)

    


    logger.info(
        "Prepared %s for Tau 1: AR(OS)=%d, SR(OS)=%d, QCD FF=%.4f",
        category_name,
        len(data_AR_OS_tau1),
        len(data_SR_OS[(data_SR_OS.process == 0)]),
        global_ff_tau1,
    )
    logger.info(
        "Prepared %s for Tau 2: AR(OS)=%d, SR(OS)=%d, QCD FF=%.4f",
        category_name,
        len(data_AR_OS_tau2),
        len(data_SR_OS[(data_SR_OS.process == 0)]),
        global_ff_tau2,
    )

    data_AR_OS_nf_tau1, data_AR_OS_nf_tau2, ar_os_clipping_mask_tau1, ar_os_clipping_mask_tau2 = normalizing_flow_ff(
        data_AR_OS_tau1,
        data_AR_OS_tau2,
        variables,
        model_AR_like_tau1,
        model_SR_like_tau1,
        model_AR_like_tau2,
        model_SR_like_tau2,
        global_ff_tau1,
        global_ff_tau2,
        device,
        plotting=True,
        plot_dir=category_plot_dir,
        include_njets=include_njets_feature,
        njets_title=njets_title
        )
    
    if args.plot_ff_values:
        plot_ff_values(data_AR_OS_nf_tau1['ff_nf_tau1'], data_AR_OS_nf_tau2['ff_nf_tau2'], category_plot_dir, njets_title)
    if not args.plot_ff_results and not args.plot_ar_data_with_clipping:
        return

    process_map = load_config(cfg_path['process_map'][args.embedding])    
    if args.embedding == 'embedding':        
        
        # ----- AR OS tau 1-----
        data_diboson_AR_OS_tau1 = data_AR_tau1[((data_AR_tau1.process == process_map["diboson_J"]) | (data_AR_tau1.process == process_map["diboson_L"]))]
        data_DY_AR_OS_tau1 = data_AR_tau1[(data_AR_tau1.process == process_map["DYjets_J"]) | (data_AR_tau1.process == process_map["DYjets_L"])]
        data_embed_AR_OS_tau1 = data_AR_tau1[(data_AR_tau1.process == process_map["embedding"])]
        data_ST_AR_OS_tau1 = data_AR_tau1[(data_AR_tau1.process == process_map["ST_J"]) | (data_AR_tau1.process == process_map["ST_L"])]
        data_ttbar_AR_OS_tau1 = data_AR_tau1[(data_AR_tau1.process == process_map["ttbar_J"]) | (data_AR_tau1.process == process_map["ttbar_L"])]
        data_wjets_AR_OS_tau1 = data_AR_tau1[(data_AR_tau1.process == process_map["Wjets"])]

        # ----- AR OS tau 2-----
        data_diboson_AR_OS_tau2 = data_AR_tau2[((data_AR_tau2.process == process_map["diboson_J"]) | (data_AR_tau2.process == process_map["diboson_L"]))]
        data_DY_AR_OS_tau2 = data_AR_tau2[(data_AR_tau2.process == process_map["DYjets_J"]) | (data_AR_tau2.process == process_map["DYjets_L"])]
        data_embed_AR_OS_tau2 = data_AR_tau2[(data_AR_tau2.process == process_map["embedding"])]
        data_ST_AR_OS_tau2 = data_AR_tau2[(data_AR_tau2.process == process_map["ST_J"]) | (data_AR_tau2.process == process_map["ST_L"])]
        data_ttbar_AR_OS_tau2 = data_AR_tau2[(data_AR_tau2.process == process_map["ttbar_J"]) | (data_AR_tau2.process == process_map["ttbar_L"])]
        data_wjets_AR_OS_tau2 = data_AR_tau2[(data_AR_tau2.process == process_map["Wjets"])]

        # ----- FF -----
        data_diboson_AR_OS_nf_tau1, data_diboson_AR_OS_nf_tau2, _, _ = normalizing_flow_ff(data_diboson_AR_OS_tau1, data_diboson_AR_OS_tau2, variables, model_AR_like_tau1, model_SR_like_tau1, model_AR_like_tau2, model_SR_like_tau2, global_ff_tau1, global_ff_tau2, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, njets_title=njets_title)
        data_DY_AR_OS_nf_tau1, data_DY_AR_OS_nf_tau2, _, _ = normalizing_flow_ff(data_DY_AR_OS_tau1, data_DY_AR_OS_tau2, variables, model_AR_like_tau1, model_SR_like_tau1, model_AR_like_tau2, model_SR_like_tau2, global_ff_tau1, global_ff_tau2, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, njets_title=njets_title)
        data_embed_AR_OS_nf_tau1, data_embed_AR_OS_nf_tau2, _, _ = normalizing_flow_ff(data_embed_AR_OS_tau1, data_embed_AR_OS_tau2, variables, model_AR_like_tau1, model_SR_like_tau1, model_AR_like_tau2, model_SR_like_tau2, global_ff_tau1, global_ff_tau2, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, njets_title=njets_title)
        data_ST_AR_OS_nf_tau1, data_ST_AR_OS_nf_tau2, _, _ = normalizing_flow_ff(data_ST_AR_OS_tau1, data_ST_AR_OS_tau2, variables, model_AR_like_tau1, model_SR_like_tau1, model_AR_like_tau2, model_SR_like_tau2, global_ff_tau1, global_ff_tau2, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, njets_title=njets_title)
        data_ttbar_AR_OS_nf_tau1, data_ttbar_AR_OS_nf_tau2, _, _ = normalizing_flow_ff(data_ttbar_AR_OS_tau1, data_ttbar_AR_OS_tau2, variables, model_AR_like_tau1, model_SR_like_tau1, model_AR_like_tau2, model_SR_like_tau2, global_ff_tau1, global_ff_tau2, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, njets_title=njets_title)
        data_wjets_AR_OS_nf_tau1, data_wjets_AR_OS_nf_tau2, _, _ = normalizing_flow_ff(data_wjets_AR_OS_tau1, data_wjets_AR_OS_tau2, variables, model_AR_like_tau1, model_SR_like_tau1, model_AR_like_tau2, model_SR_like_tau2, global_ff_tau1, global_ff_tau2, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, njets_title=njets_title)

        # ----- SR OS -----
        data_events = data_SR_OS[(data_SR_OS.process == process_map["data"])]
        data_diboson_SR_OS = data_SR_OS[(data_SR_OS.process == process_map["diboson_J"]) | (data_SR_OS.process == process_map["diboson_L"])]
        data_DY_SR_OS = data_SR_OS[(data_SR_OS.process == process_map["DYjets_J"]) | (data_SR_OS.process == process_map["DYjets_L"])]
        data_embed_SR_OS = data_SR_OS[(data_SR_OS.process == process_map["embedding"])]
        data_ST_SR_OS = data_SR_OS[(data_SR_OS.process == process_map["ST_J"]) | (data_SR_OS.process == process_map["ST_L"])]
        data_ttbar_SR_OS = data_SR_OS[(data_SR_OS.process == process_map["ttbar_J"]) | (data_SR_OS.process == process_map["ttbar_L"])]
        data_wjets_SR_OS = data_SR_OS[(data_SR_OS.process == process_map["Wjets"])]

        total_variables = len(list_variables)
        for index, (var, bins, xlabel) in enumerate(zip(list_variables, list_bins, list_xlabels), start=1):
            if should_log_plot_progress(index, total_variables, 1):
                logger.info(
                    "Plotting %s: %d/%d variables (%s)",
                    category_name,
                    index,
                    total_variables,
                    var,
                )
            
            # Plot AR data with clipping information if requested
            if args.plot_ar_data_with_clipping:
                plot_ar_data_with_clipping_info(
                    var=var,
                    bins=bins,
                    xlabel=xlabel,
                    data_ar_os_full=data_AR_OS_tau1,
                    clipping_mask=ar_os_clipping_mask_tau1,
                    njets_title=njets_title,
                    output_dir=category_plot_dir,
                    tau_label='tau_1',
                )
                plot_ar_data_with_clipping_info(
                    var=var,
                    bins=bins,
                    xlabel=xlabel,
                    data_ar_os_full=data_AR_OS_tau2,
                    clipping_mask=ar_os_clipping_mask_tau2,
                    njets_title=njets_title,
                    output_dir=category_plot_dir,
                    tau_label='tau_2',
                )
            
            if not args.plot_ff_results:
                continue

            # ----- counts FF tau1 -----
            counts_ff_data_tau1, bin_edges = np.histogram(data_AR_OS_nf_tau1[var], weights=data_AR_OS_nf_tau1['ff_nf_tau1'], bins=bins)
            counts_ff_data2_tau1, _ = np.histogram(data_AR_OS_nf_tau1[var], weights=data_AR_OS_nf_tau1['ff_nf_tau1']**2, bins=bins)

            counts_ff_diboson_tau1, _ = np.histogram(data_diboson_AR_OS_nf_tau1[var], weights=data_diboson_AR_OS_nf_tau1.weight * data_diboson_AR_OS_nf_tau1['ff_nf_tau1'], bins=bins)
            counts_ff_diboson2_tau1, _ = np.histogram(data_diboson_AR_OS_nf_tau1[var], weights=data_diboson_AR_OS_nf_tau1.weight**2, bins=bins)# * data_diboson_AR_OS_nf['ff_nf'])**2, bins=bins)
            counts_ff_DY_tau1, _ = np.histogram(data_DY_AR_OS_nf_tau1[var], weights=data_DY_AR_OS_nf_tau1.weight * data_DY_AR_OS_nf_tau1['ff_nf_tau1'], bins=bins)
            counts_ff_DY2_tau1, _ = np.histogram(data_DY_AR_OS_nf_tau1[var], weights=(data_DY_AR_OS_nf_tau1.weight * data_DY_AR_OS_nf_tau1['ff_nf_tau1'])**2, bins=bins)
            counts_ff_embed_tau1,_ = np.histogram(data_embed_AR_OS_nf_tau1[var], weights=data_embed_AR_OS_nf_tau1.weight * data_embed_AR_OS_nf_tau1['ff_nf_tau1'], bins=bins)
            counts_ff_embed2_tau1, _ = np.histogram(data_embed_AR_OS_nf_tau1[var], weights=(data_embed_AR_OS_nf_tau1.weight * data_embed_AR_OS_nf_tau1['ff_nf_tau1'])**2, bins=bins)
            counts_ff_ST_tau1, _ = np.histogram(data_ST_AR_OS_nf_tau1[var], weights=data_ST_AR_OS_nf_tau1.weight * data_ST_AR_OS_nf_tau1['ff_nf_tau1'], bins=bins)
            counts_ff_ST2_tau1, _ = np.histogram(data_ST_AR_OS_nf_tau1[var], weights=(data_ST_AR_OS_nf_tau1.weight * data_ST_AR_OS_nf_tau1['ff_nf_tau1'])**2, bins=bins)
            counts_ff_ttbar_tau1, _ = np.histogram(data_ttbar_AR_OS_nf_tau1[var], weights=data_ttbar_AR_OS_nf_tau1.weight * data_ttbar_AR_OS_nf_tau1['ff_nf_tau1'], bins=bins)
            counts_ff_ttbar2_tau1, _ = np.histogram(data_ttbar_AR_OS_nf_tau1[var], weights=(data_ttbar_AR_OS_nf_tau1.weight * data_ttbar_AR_OS_nf_tau1['ff_nf_tau1'])**2, bins=bins)
            counts_ff_wjets_tau1, _ = np.histogram(data_wjets_AR_OS_nf_tau1[var], weights=data_wjets_AR_OS_nf_tau1.weight * data_wjets_AR_OS_nf_tau1['ff_nf_tau1'], bins=bins)
            counts_ff_wjets2_tau1, _ = np.histogram(data_wjets_AR_OS_nf_tau1[var], weights=(data_wjets_AR_OS_nf_tau1.weight * data_wjets_AR_OS_nf_tau1['ff_nf_tau1'])**2, bins=bins)
            counts_FF_tau1 = counts_ff_data_tau1 - counts_ff_diboson_tau1 - counts_ff_DY_tau1 - counts_ff_embed_tau1 - counts_ff_ST_tau1 - counts_ff_ttbar_tau1 - counts_ff_wjets_tau1

            # ----- counts FF tau2 -----
            counts_ff_data_tau2, bin_edges = np.histogram(data_AR_OS_nf_tau2[var], weights=data_AR_OS_nf_tau2['ff_nf_tau2'], bins=bins)
            counts_ff_data2_tau2, _ = np.histogram(data_AR_OS_nf_tau2[var], weights=data_AR_OS_nf_tau2['ff_nf_tau2']**2, bins=bins)

            counts_ff_diboson_tau2, _ = np.histogram(data_diboson_AR_OS_nf_tau2[var], weights=data_diboson_AR_OS_nf_tau2.weight * data_diboson_AR_OS_nf_tau2['ff_nf_tau2'], bins=bins)
            counts_ff_diboson2_tau2, _ = np.histogram(data_diboson_AR_OS_nf_tau2[var], weights=(data_diboson_AR_OS_nf_tau2.weight * data_diboson_AR_OS_nf_tau2['ff_nf_tau2'])**2, bins=bins)
            counts_ff_DY_tau2, _ = np.histogram(data_DY_AR_OS_nf_tau2[var], weights=data_DY_AR_OS_nf_tau2.weight * data_DY_AR_OS_nf_tau2['ff_nf_tau2'], bins=bins)
            counts_ff_DY2_tau2, _ = np.histogram(data_DY_AR_OS_nf_tau2[var], weights=(data_DY_AR_OS_nf_tau2.weight * data_DY_AR_OS_nf_tau2['ff_nf_tau2'])**2, bins=bins)
            counts_ff_embed_tau2,_ = np.histogram(data_embed_AR_OS_nf_tau2[var], weights=data_embed_AR_OS_nf_tau2.weight * data_embed_AR_OS_nf_tau2['ff_nf_tau2'], bins=bins)
            counts_ff_embed2_tau2, _ = np.histogram(data_embed_AR_OS_nf_tau2[var], weights=(data_embed_AR_OS_nf_tau2.weight * data_embed_AR_OS_nf_tau2['ff_nf_tau2'])**2, bins=bins)
            counts_ff_ST_tau2, _ = np.histogram(data_ST_AR_OS_nf_tau2[var], weights=data_ST_AR_OS_nf_tau2.weight * data_ST_AR_OS_nf_tau2['ff_nf_tau2'], bins=bins)
            counts_ff_ST2_tau2, _ = np.histogram(data_ST_AR_OS_nf_tau2[var], weights=(data_ST_AR_OS_nf_tau2.weight * data_ST_AR_OS_nf_tau2['ff_nf_tau2'])**2, bins=bins)
            counts_ff_ttbar_tau2, _ = np.histogram(data_ttbar_AR_OS_nf_tau2[var], weights=data_ttbar_AR_OS_nf_tau2.weight * data_ttbar_AR_OS_nf_tau2['ff_nf_tau2'], bins=bins)
            counts_ff_ttbar2_tau2, _ = np.histogram(data_ttbar_AR_OS_nf_tau2[var], weights=(data_ttbar_AR_OS_nf_tau2.weight * data_ttbar_AR_OS_nf_tau2['ff_nf_tau2'])**2, bins=bins)
            counts_ff_wjets_tau2, _ = np.histogram(data_wjets_AR_OS_nf_tau2[var], weights=data_wjets_AR_OS_nf_tau2.weight * data_wjets_AR_OS_nf_tau2['ff_nf_tau2'], bins=bins)
            counts_ff_wjets2_tau2, _ = np.histogram(data_wjets_AR_OS_nf_tau2[var], weights=(data_wjets_AR_OS_nf_tau2.weight * data_wjets_AR_OS_nf_tau2['ff_nf_tau2'])**2, bins=bins)
            counts_FF_tau2 = counts_ff_data_tau2 - counts_ff_diboson_tau2 - counts_ff_DY_tau2 - counts_ff_embed_tau2 - counts_ff_ST_tau2 - counts_ff_ttbar_tau2 -counts_ff_wjets_tau2

            # ----- total FF, with factor 0.5 assuming both FF are weighted equally -----
            
            #rat = len(data_AR_OS_nf_tau1[var])/(len(data_AR_OS_nf_tau1[var])+len(data_AR_OS_nf_tau2[var]))
            #counts_FF = rat*counts_FF_tau1 + (1-rat)*counts_FF_tau2

            counts_FF = 0.5*(counts_FF_tau1 + counts_FF_tau2)

            # ----- counts SR -----
            counts_diboson, _ = np.histogram(data_diboson_SR_OS[var], weights=data_diboson_SR_OS.weight, bins=bins)
            counts_diboson2, _ = np.histogram(data_diboson_SR_OS[var], weights=data_diboson_SR_OS.weight**2, bins=bins)
            counts_DY, _ = np.histogram(data_DY_SR_OS[var], weights=data_DY_SR_OS.weight, bins=bins)
            counts_DY2, _ = np.histogram(data_DY_SR_OS[var], weights=data_DY_SR_OS.weight**2, bins=bins)
            counts_embed, _ = np.histogram(data_embed_SR_OS[var], weights=data_embed_SR_OS.weight, bins=bins)
            counts_embed2, _ = np.histogram(data_embed_SR_OS[var], weights=data_embed_SR_OS.weight**2, bins=bins)
            counts_ST, _ = np.histogram(data_ST_SR_OS[var], weights=data_ST_SR_OS.weight, bins=bins)
            counts_ST2, _ = np.histogram(data_ST_SR_OS[var], weights=data_ST_SR_OS.weight**2, bins=bins)
            counts_ttbar, _ = np.histogram(data_ttbar_SR_OS[var], weights=data_ttbar_SR_OS.weight, bins=bins)
            counts_ttbar2, _ = np.histogram(data_ttbar_SR_OS[var], weights=data_ttbar_SR_OS.weight**2, bins=bins)
            counts_wjets, _ = np.histogram(data_wjets_SR_OS[var], weights=data_wjets_SR_OS.weight, bins=bins)
            counts_wjets2, _ = np.histogram(data_wjets_SR_OS[var], weights=data_wjets_SR_OS.weight**2, bins=bins)
            
            counts_data, _ = np.histogram(data_events[var], bins=bins)

            bin_widths = np.diff(bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            fig, ax = plt.subplots(
                2, 1,
                figsize=(9,9),
                sharex=True,
                gridspec_kw={'height_ratios': [4,1], 'hspace': 0.05},
                constrained_layout=True
            )

            CMS_CHANNEL_TITLE(ax)
            CMS_LUMI_TITLE(ax)
            CMS_LABEL(ax)
            CMS_NJETS_TITLE(ax, title=njets_title)

            y_error = np.sqrt(counts_data)
            x_error = 0.5 * bin_widths
            num = np.sqrt(
                counts_ff_data2_tau1 + counts_ff_data2_tau2 +
                counts_ff_diboson2_tau1 + counts_ff_diboson2_tau2 +
                counts_ff_ttbar2_tau1 + counts_ff_ttbar2_tau2 +
                counts_ff_ST2_tau1 + counts_ff_ST2_tau2 +
                counts_ff_DY2_tau1 + counts_ff_DY2_tau2 +
                counts_ff_embed2_tau1 + counts_ff_embed2_tau2 +
                counts_ff_wjets2_tau1 + counts_ff_wjets2_tau2 +
                counts_diboson2 + counts_ttbar2 + counts_DY2 + counts_ST2 + counts_embed2 + counts_wjets2
            )

            den = (
                counts_FF + counts_diboson + 
                counts_ttbar + counts_ST + counts_DY + counts_embed + counts_wjets
            )

            y_error_stat = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
        
            stack_components = [
                (counts_diboson, "#94a4a2", 'Diboson'),
                (counts_ttbar, '#832db6', r'$t\bar{t} \to \tau$'),
                (counts_ST, "#717581", r"Single t"),
                (counts_DY, '#3f90da', r'$Z \to \ell \ell$'),
                (counts_wjets, '#e76300', r"W+jets"),
                (counts_embed, '#ffa90e', r'$\tau$ embedded'),
                (counts_FF, "#a96b59", r'Jet $\rightarrow \tau_h$'),
            ]
            counts_stack_total = draw_stacked_stepfill(ax[0], bin_edges, stack_components)
            ax[0].stairs(counts_stack_total, bin_edges, color='black', linewidth=0.7)

            ax[0].errorbar(bin_centers, counts_data, yerr=y_error, xerr=x_error, fmt='o', color='black', label='Data', markersize=6, elinewidth=1.2, capsize=0)
            ax[0].set_ylabel("Events", fontsize=23)
            handles, labels = ax[0].get_legend_handles_labels()
            handles = handles[::-1]
            labels = labels[::-1]
            handles, labels = reorder_for_rowwise_legend(handles, labels, ncol=3)
            ax[0].legend(handles, labels, title=' ', loc='upper right', ncol=3, frameon=False, fontsize='x-small')
            adjust_ylim_for_legend(ax[0])
            ax[0].tick_params(direction='in', top=True, right=True)
            #ax[0].ticklabel_format(style='sci', axis='y', scilimits=(3,3))

            ax[1].errorbar(
                bin_centers,
                np.divide(counts_data, den, out=np.zeros_like(counts_data, dtype=float), where=den != 0),
                xerr=x_error,
                yerr=np.divide(y_error, counts_data, out=np.zeros_like(counts_data, dtype=float), where=counts_data != 0),
                fmt='o',
                color='black',
                markersize=6,
                label=(r'NF $F_\text{F}$')
            )
            ax[1].fill_between(bin_centers, 1 - y_error_stat, 1 + y_error_stat, color="gray", alpha=0.3, step='mid', label="Stat. Unc.")
            ax[1].axhline(1, color='red', linestyle='--', linewidth=1.5)
            ax[1].set_ylabel("Data / Model", fontsize=23, loc='center')
            ax[1].set_ylim([args.ratio_ylim_min, args.ratio_ylim_max])
            ax[1].grid(True, linestyle=':', alpha=0.7)
            ax[1].tick_params(direction='in', top=True, right=True)
            ax[1].legend(loc='upper left', ncol=2, frameon=False, fontsize='xx-small') #, bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0
            
           
            ax[-1].set_xlabel(xlabel)
            fig.savefig(category_plot_dir / f'{var}.png')
            fig.savefig(category_plot_dir / f'{var}.pdf')
            plt.close(fig)

    elif args.embedding == 'no_embedding':
        
        # ----- AR OS tau 1-----
        data_diboson_AR_OS_tau1 = data_AR_tau1[((data_AR_tau1.process == process_map["diboson_J"]) | (data_AR_tau1.process == process_map["diboson_L"]) | (data_AR_tau1.process == process_map["diboson_T"]))]
        data_DY_AR_OS_tau1 = data_AR_tau1[(data_AR_tau1.process == process_map["DYjets_J"]) | (data_AR_tau1.process == process_map["DYjets_L"]) | (data_AR_tau1.process == process_map["DYjets_T"])]
        data_ST_AR_OS_tau1 = data_AR_tau1[(data_AR_tau1.process == process_map["ST_J"]) | (data_AR_tau1.process == process_map["ST_L"]) | (data_AR_tau1.process == process_map["ST_T"])]
        data_ttbar_AR_OS_tau1 = data_AR_tau1[(data_AR_tau1.process == process_map["ttbar_J"]) | (data_AR_tau1.process == process_map["ttbar_L"]) | (data_AR_tau1.process == process_map["ttbar_T"])]
        data_wjets_AR_OS_tau1 = data_AR_tau1[(data_AR_tau1.process == process_map["Wjets"])]

        # ----- AR OS tau 2-----
        data_diboson_AR_OS_tau2 = data_AR_tau2[((data_AR_tau2.process == process_map["diboson_J"]) | (data_AR_tau2.process == process_map["diboson_L"]) | (data_AR_tau2.process == process_map["diboson_T"]))]
        data_DY_AR_OS_tau2 = data_AR_tau2[(data_AR_tau2.process == process_map["DYjets_J"]) | (data_AR_tau2.process == process_map["DYjets_L"]) | (data_AR_tau2.process == process_map["DYjets_T"])]
        data_ST_AR_OS_tau2 = data_AR_tau2[(data_AR_tau2.process == process_map["ST_J"]) | (data_AR_tau2.process == process_map["ST_L"]) | (data_AR_tau2.process == process_map["ST_T"])]
        data_ttbar_AR_OS_tau2 = data_AR_tau2[(data_AR_tau2.process == process_map["ttbar_J"]) | (data_AR_tau2.process == process_map["ttbar_L"]) | (data_AR_tau2.process == process_map["ttbar_T"])]
        data_wjets_AR_OS_tau2 = data_AR_tau2[(data_AR_tau2.process == process_map["Wjets"])]

        # ----- FF -----
        data_diboson_AR_OS_nf_tau1, data_diboson_AR_OS_nf_tau2, _, _ = normalizing_flow_ff(data_diboson_AR_OS_tau1, data_diboson_AR_OS_tau2, variables, model_AR_like_tau1, model_SR_like_tau1, model_AR_like_tau2, model_SR_like_tau2, global_ff_tau1, global_ff_tau2, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, njets_title=njets_title)
        data_DY_AR_OS_nf_tau1, data_DY_AR_OS_nf_tau2, _, _ = normalizing_flow_ff(data_DY_AR_OS_tau1, data_DY_AR_OS_tau2, variables, model_AR_like_tau1, model_SR_like_tau1, model_AR_like_tau2, model_SR_like_tau2, global_ff_tau1, global_ff_tau2, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, njets_title=njets_title)
        data_ST_AR_OS_nf_tau1, data_ST_AR_OS_nf_tau2, _, _ = normalizing_flow_ff(data_ST_AR_OS_tau1, data_ST_AR_OS_tau2, variables, model_AR_like_tau1, model_SR_like_tau1, model_AR_like_tau2, model_SR_like_tau2, global_ff_tau1, global_ff_tau2, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, njets_title=njets_title)
        data_ttbar_AR_OS_nf_tau1, data_ttbar_AR_OS_nf_tau2, _, _ = normalizing_flow_ff(data_ttbar_AR_OS_tau1, data_ttbar_AR_OS_tau2, variables, model_AR_like_tau1, model_SR_like_tau1, model_AR_like_tau2, model_SR_like_tau2, global_ff_tau1, global_ff_tau2, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, njets_title=njets_title)
        data_wjets_AR_OS_nf_tau1, data_wjets_AR_OS_nf_tau2, _, _ = normalizing_flow_ff(data_wjets_AR_OS_tau1, data_wjets_AR_OS_tau2, variables, model_AR_like_tau1, model_SR_like_tau1, model_AR_like_tau2, model_SR_like_tau2, global_ff_tau1, global_ff_tau2, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, njets_title=njets_title)

        # ----- SR OS -----
        data_events = data_SR_OS[(data_SR_OS.process == process_map["data"])]
        data_diboson_SR_OS = data_SR_OS[(data_SR_OS.process == process_map["diboson_J"]) | (data_SR_OS.process == process_map["diboson_L"]) | (data_SR_OS.process == process_map["diboson_T"])]
        data_DY_SR_OS = data_SR_OS[(data_SR_OS.process == process_map["DYjets_J"]) | (data_SR_OS.process == process_map["DYjets_L"]) | (data_SR_OS.process == process_map["DYjets_T"])]
        data_ST_SR_OS = data_SR_OS[(data_SR_OS.process == process_map["ST_J"]) | (data_SR_OS.process == process_map["ST_L"]) | (data_SR_OS.process == process_map["ST_T"])]
        data_ttbar_SR_OS = data_SR_OS[(data_SR_OS.process == process_map["ttbar_J"]) | (data_SR_OS.process == process_map["ttbar_L"]) | (data_SR_OS.process == process_map["ttbar_T"])]
        data_wjets_SR_OS = data_SR_OS[(data_SR_OS.process == process_map["Wjets"])]
    
        total_variables = len(list_variables)
        for index, (var, bins, xlabel) in enumerate(zip(list_variables, list_bins, list_xlabels), start=1):
            if should_log_plot_progress(index, total_variables, 1):
                logger.info(
                    "Plotting %s: %d/%d variables (%s)",
                    category_name,
                    index,
                    total_variables,
                    var,
                )
            
            # Plot AR data with clipping information if requested
            if args.plot_ar_data_with_clipping:
                plot_ar_data_with_clipping_info(
                    var=var,
                    bins=bins,
                    xlabel=xlabel,
                    data_ar_os_full=data_AR_OS_tau1,
                    clipping_mask=ar_os_clipping_mask_tau1,
                    njets_title=njets_title,
                    output_dir=category_plot_dir,
                    tau_label='tau_1',
                )
                plot_ar_data_with_clipping_info(
                    var=var,
                    bins=bins,
                    xlabel=xlabel,
                    data_ar_os_full=data_AR_OS_tau2,
                    clipping_mask=ar_os_clipping_mask_tau2,
                    njets_title=njets_title,
                    output_dir=category_plot_dir,
                    tau_label='tau_2',
                )
            
            if not args.plot_ff_results:
                continue

            
            # ----- counts FF tau1 -----
            counts_ff_data_tau1, bin_edges = np.histogram(data_AR_OS_nf_tau1[var], weights=data_AR_OS_nf_tau1['ff_nf_tau1'], bins=bins)
            counts_ff_data2_tau1, _ = np.histogram(data_AR_OS_nf_tau1[var], weights=data_AR_OS_nf_tau1['ff_nf_tau1']**2, bins=bins)

            counts_ff_diboson_tau1, _ = np.histogram(data_diboson_AR_OS_nf_tau1[var], weights=data_diboson_AR_OS_nf_tau1.weight * data_diboson_AR_OS_nf_tau1['ff_nf_tau1'], bins=bins)
            counts_ff_diboson2_tau1, _ = np.histogram(data_diboson_AR_OS_nf_tau1[var], weights=data_diboson_AR_OS_nf_tau1.weight**2, bins=bins)# * data_diboson_AR_OS_nf['ff_nf'])**2, bins=bins)
            counts_ff_DY_tau1, _ = np.histogram(data_DY_AR_OS_nf_tau1[var], weights=data_DY_AR_OS_nf_tau1.weight * data_DY_AR_OS_nf_tau1['ff_nf_tau1'], bins=bins)
            counts_ff_DY2_tau1, _ = np.histogram(data_DY_AR_OS_nf_tau1[var], weights=(data_DY_AR_OS_nf_tau1.weight * data_DY_AR_OS_nf_tau1['ff_nf_tau1'])**2, bins=bins)
            counts_ff_ST_tau1, _ = np.histogram(data_ST_AR_OS_nf_tau1[var], weights=data_ST_AR_OS_nf_tau1.weight * data_ST_AR_OS_nf_tau1['ff_nf_tau1'], bins=bins)
            counts_ff_ST2_tau1, _ = np.histogram(data_ST_AR_OS_nf_tau1[var], weights=(data_ST_AR_OS_nf_tau1.weight * data_ST_AR_OS_nf_tau1['ff_nf_tau1'])**2, bins=bins)
            counts_ff_ttbar_tau1, _ = np.histogram(data_ttbar_AR_OS_nf_tau1[var], weights=data_ttbar_AR_OS_nf_tau1.weight * data_ttbar_AR_OS_nf_tau1['ff_nf_tau1'], bins=bins)
            counts_ff_ttbar2_tau1, _ = np.histogram(data_ttbar_AR_OS_nf_tau1[var], weights=(data_ttbar_AR_OS_nf_tau1.weight * data_ttbar_AR_OS_nf_tau1['ff_nf_tau1'])**2, bins=bins)
            counts_ff_wjets_tau1, _ = np.histogram(data_wjets_AR_OS_nf_tau1[var], weights=data_wjets_AR_OS_nf_tau1.weight * data_wjets_AR_OS_nf_tau1['ff_nf_tau1'], bins=bins)
            counts_ff_wjets2_tau1, _ = np.histogram(data_wjets_AR_OS_nf_tau1[var], weights=(data_wjets_AR_OS_nf_tau1.weight * data_wjets_AR_OS_nf_tau1['ff_nf_tau1'])**2, bins=bins)
            counts_FF_tau1 = counts_ff_data_tau1 - counts_ff_diboson_tau1 - counts_ff_DY_tau1 - counts_ff_ST_tau1 - counts_ff_ttbar_tau1 - counts_ff_wjets_tau1

            # ----- counts FF tau2 -----
            counts_ff_data_tau2, bin_edges = np.histogram(data_AR_OS_nf_tau2[var], weights=data_AR_OS_nf_tau2['ff_nf_tau2'], bins=bins)
            counts_ff_data2_tau2, _ = np.histogram(data_AR_OS_nf_tau2[var], weights=data_AR_OS_nf_tau2['ff_nf_tau2']**2, bins=bins)

            counts_ff_diboson_tau2, _ = np.histogram(data_diboson_AR_OS_nf_tau2[var], weights=data_diboson_AR_OS_nf_tau2.weight * data_diboson_AR_OS_nf_tau2['ff_nf_tau2'], bins=bins)
            counts_ff_diboson2_tau2, _ = np.histogram(data_diboson_AR_OS_nf_tau2[var], weights=(data_diboson_AR_OS_nf_tau2.weight * data_diboson_AR_OS_nf_tau2['ff_nf_tau2'])**2, bins=bins)
            counts_ff_DY_tau2, _ = np.histogram(data_DY_AR_OS_nf_tau2[var], weights=data_DY_AR_OS_nf_tau2.weight * data_DY_AR_OS_nf_tau2['ff_nf_tau2'], bins=bins)
            counts_ff_DY2_tau2, _ = np.histogram(data_DY_AR_OS_nf_tau2[var], weights=(data_DY_AR_OS_nf_tau2.weight * data_DY_AR_OS_nf_tau2['ff_nf_tau2'])**2, bins=bins)
            counts_ff_ST_tau2, _ = np.histogram(data_ST_AR_OS_nf_tau2[var], weights=data_ST_AR_OS_nf_tau2.weight * data_ST_AR_OS_nf_tau2['ff_nf_tau2'], bins=bins)
            counts_ff_ST2_tau2, _ = np.histogram(data_ST_AR_OS_nf_tau2[var], weights=(data_ST_AR_OS_nf_tau2.weight * data_ST_AR_OS_nf_tau2['ff_nf_tau2'])**2, bins=bins)
            counts_ff_ttbar_tau2, _ = np.histogram(data_ttbar_AR_OS_nf_tau2[var], weights=data_ttbar_AR_OS_nf_tau2.weight * data_ttbar_AR_OS_nf_tau2['ff_nf_tau2'], bins=bins)
            counts_ff_ttbar2_tau2, _ = np.histogram(data_ttbar_AR_OS_nf_tau2[var], weights=(data_ttbar_AR_OS_nf_tau2.weight * data_ttbar_AR_OS_nf_tau2['ff_nf_tau2'])**2, bins=bins)
            counts_ff_wjets_tau2, _ = np.histogram(data_wjets_AR_OS_nf_tau2[var], weights=data_wjets_AR_OS_nf_tau2.weight * data_wjets_AR_OS_nf_tau2['ff_nf_tau2'], bins=bins)
            counts_ff_wjets2_tau2, _ = np.histogram(data_wjets_AR_OS_nf_tau2[var], weights=(data_wjets_AR_OS_nf_tau2.weight * data_wjets_AR_OS_nf_tau2['ff_nf_tau2'])**2, bins=bins)
            counts_FF_tau2 = counts_ff_data_tau2 - counts_ff_diboson_tau2 - counts_ff_DY_tau2 - counts_ff_ST_tau2 - counts_ff_ttbar_tau2 - counts_ff_wjets_tau2

            # ----- total FF, with factor 0.5 assuming both FF are weighted equally -----
            counts_FF = 0.5*(counts_FF_tau1 + counts_FF_tau2)

            # ----- counts SR -----
            counts_diboson, _ = np.histogram(data_diboson_SR_OS[var], weights=data_diboson_SR_OS.weight, bins=bins)
            counts_diboson2, _ = np.histogram(data_diboson_SR_OS[var], weights=data_diboson_SR_OS.weight**2, bins=bins)
            counts_DY, _ = np.histogram(data_DY_SR_OS[var], weights=data_DY_SR_OS.weight, bins=bins)
            counts_DY2, _ = np.histogram(data_DY_SR_OS[var], weights=data_DY_SR_OS.weight**2, bins=bins)
            counts_ST, _ = np.histogram(data_ST_SR_OS[var], weights=data_ST_SR_OS.weight, bins=bins)
            counts_ST2, _ = np.histogram(data_ST_SR_OS[var], weights=data_ST_SR_OS.weight**2, bins=bins)
            counts_ttbar, _ = np.histogram(data_ttbar_SR_OS[var], weights=data_ttbar_SR_OS.weight, bins=bins)
            counts_ttbar2, _ = np.histogram(data_ttbar_SR_OS[var], weights=data_ttbar_SR_OS.weight**2, bins=bins)
            counts_wjets, _ = np.histogram(data_wjets_SR_OS[var], weights=data_wjets_SR_OS.weight, bins=bins)
            counts_wjets2, _ = np.histogram(data_wjets_SR_OS[var], weights=data_wjets_SR_OS.weight**2, bins=bins)
            
            counts_data, _ = np.histogram(data_events[var], bins=bins)

            bin_widths = np.diff(bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            fig, ax = plt.subplots(
                2, 1,
                figsize=(9,9),
                sharex=True,
                gridspec_kw={'height_ratios': [4,1], 'hspace': 0.05},
                constrained_layout=True
            )

            CMS_CHANNEL_TITLE(ax)
            CMS_LUMI_TITLE(ax)
            CMS_LABEL(ax)
            CMS_NJETS_TITLE(ax, title=njets_title)

            y_error = np.sqrt(counts_data)
            x_error = 0.5 * bin_widths
            num = np.sqrt(
                counts_ff_data2_tau1 + counts_ff_data2_tau2 +
                counts_ff_diboson2_tau1 + counts_ff_diboson2_tau2 +
                counts_ff_ttbar2_tau1 + counts_ff_ttbar2_tau2 +
                counts_ff_ST2_tau1 + counts_ff_ST2_tau2 +
                counts_ff_DY2_tau1 + counts_ff_DY2_tau2 +
                counts_ff_wjets2_tau1 + counts_ff_wjets2_tau2 +
                counts_diboson2 + counts_ttbar2 + counts_DY2 + counts_ST2 + counts_wjets2
            )

            den = (
                counts_FF + counts_diboson + 
                counts_ttbar + counts_ST + counts_DY + counts_wjets
            )

            y_error_stat = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
        
            
            stack_components = [
                (counts_diboson, "#94a4a2", 'Diboson'),
                (counts_ttbar, '#832db6', r'$t\bar{t} \to \tau$'),
                (counts_ST, "#717581", r"Single t"),
                (counts_wjets, '#e76300', r"W+jets"),
                (counts_DY, '#3f90da', r'$Z \to \ell \ell$'),
                (counts_FF, "#a96b59", r'Jet $\rightarrow \tau_h$'),
            ]
            counts_stack_total = draw_stacked_stepfill(ax[0], bin_edges, stack_components)
            ax[0].stairs(counts_stack_total, bin_edges, color='black', linewidth=0.7)

            ax[0].errorbar(bin_centers, counts_data, yerr=y_error, xerr=x_error, fmt='o', color='black', label='Data', markersize=6, elinewidth=1.2, capsize=0)
            ax[0].set_ylabel("Events", fontsize=23)
            handles, labels = ax[0].get_legend_handles_labels()
            handles = handles[::-1]
            labels = labels[::-1]
            handles, labels = reorder_for_rowwise_legend(handles, labels, ncol=3)
            ax[0].legend(handles, labels, title=' ', loc='upper right', ncol=3, frameon=False, fontsize='x-small')
            adjust_ylim_for_legend(ax[0])
            ax[0].tick_params(direction='in', top=True, right=True)
            #ax[0].ticklabel_format(style='sci', axis='y', scilimits=(3,3))

            ax[1].errorbar(
                bin_centers,
                np.divide(counts_data, den, out=np.zeros_like(counts_data, dtype=float), where=den != 0),
                xerr=x_error,
                yerr=np.divide(y_error, counts_data, out=np.zeros_like(counts_data, dtype=float), where=counts_data != 0),
                fmt='o',
                color='black',
                markersize=6,
                label=(r'NF $F_\text{F}$')
            )
            ax[1].fill_between(bin_centers, 1 - y_error_stat, 1 + y_error_stat, color="gray", alpha=0.3, step='mid', label="Stat. Unc.")
            ax[1].axhline(1, color='red', linestyle='--', linewidth=1.5)
            ax[1].set_ylabel("Data / Model", fontsize=23, loc='center')
            ax[1].set_ylim([args.ratio_ylim_min, args.ratio_ylim_max])
            ax[1].grid(True, linestyle=':', alpha=0.7)
            ax[1].tick_params(direction='in', top=True, right=True)
            ax[1].legend(loc='upper left', ncol=2, frameon=False, fontsize='xx-small') #, bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0
            
            ax[-1].set_xlabel(xlabel)
            fig.savefig(category_plot_dir / f'{var}.png')
            fig.savefig(category_plot_dir / f'{var}.pdf')
            plt.close(fig)

            
    else:
        raise ValueError(f"Invalid taus configuration: {args.embedding}. Expected embedding or no_embedding.")

    logger.info("Finished %s: saved plots to %s", category_name, category_plot_dir)


def run_taylor_plots_if_requested() -> None:
    if not args.plot_taylor_coefficients:
        return

    plot_nf_taylor_analysis(plot_root_dir / 'taylor_analysis')
    plot_nf_taylor_analysis_output(plot_root_dir / 'taylor_analysis')
    plot_nf_second_order_covariance(plot_root_dir / 'taylor_analysis')


def run_all_njets_categories() -> None:
    njets_categories = [
        ('njets_0', r'$\mathrm{N_{jets}} = 0$'),
        ('njets_1', r'$\mathrm{N_{jets}} = 1$'),
        ('njets_geq_2', r'$\mathrm{N_{jets}} \geq 2$'),
        ('njets_inclusive', r'$\mathrm{N_{jets}} \geq 0$'),
    ]
    
    for category_name, njets_title in njets_categories:
        logger.info(f"Queueing plot production for {category_name}")
        if args.plot_ff_results or args.plot_nf_sampling or args.plot_ar_data_with_clipping or args.plot_ff_values:
            run_plots_for_njets_category(category_name, njets_title)


def main() -> None:
    t.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Step 1: initialize runtime context (args, models, data, labels, bins, output dirs)
    logger.info('Step 1/4: Initializing runtime context')
    initialize_runtime_context()

    # Step 2: optional model interpretability diagnostics (Taylor analysis)
    print(" ")
    logger.info('Step 2/4: Running optional Taylor diagnostics')
    run_taylor_plots_if_requested()

    # Step 3: produce category-wise plots (0, 1, >=2, inclusive)
    print(" ")
    logger.info('Step 3/4: Producing njets-category plots')
    run_all_njets_categories()

    # Step 4: finalize
    print(" ")
    logger.info('Step 4/4: Completed all njets plot categories')


if __name__ == '__main__':
    main()