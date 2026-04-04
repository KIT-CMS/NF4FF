from __future__ import annotations
import logging
import hashlib
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Tuple, Iterable
from copy import deepcopy
import pandas as pd
import numpy as np
import torch
import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import yaml
from tap import Tap
from CustomLogging import setup_logging
from classes.NeuralNetworks import RealNVP, RealNVP_NN, AffineCoupling, MLP, ConditionalRealNVP, BinaryClassifier
import correctionlib as cr
from classes.Dataclasses import ModelConfig
from classes.Collection import load_model_config, load_flow, load_conditional_flow, evaluate_pdf, compute_eventwise_fake_factors, get_my_data_qcd, get_my_data_wjets
from classes.Collection import evaluate_density_ratio_binary_classifier, compute_eventwise_fake_factors_binary_classifier
from classes.Collection import load_config, load_grouped_wjets_njets_router, load_grouped_qcd_njets_router
from classes.Plotting import CMS_CHANNEL_TITLE, CMS_LABEL, CMS_LUMI_TITLE, CMS_NJETS_TITLE, reorder_for_rowwise_legend, adjust_ylim_for_legend

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

class Args(Tap):
    model_mode: Literal['grouped_njets_split', 'single_nf', 'conditional_nf'] = 'single_nf'  # Training mode to load: grouped NF split by njets, single inclusive NF, or conditional NF with njets as input.
    ff_estimator: Literal['nf', 'binary_classifier'] = 'nf'  # FF backend: use NF models or SR/AR binary-classifier models.
    classifier_training_tag: str = ''  # Optional classifier training folder suffix after 'training_'. Empty -> pick most recent.
    classifier_hidden_layers: int = 2  # Binary-classifier selection helper: pick the most recent training with this number of hidden layers.
    apply_wjets_binary_correction: bool = True  # Apply Wjets event-wise antiDR/DR correction in binary-classifier mode.
    classifier_corrections_training_tag: str = ''  # Optional Wjets correction training folder suffix in binary_classifier_corrections. Empty -> pick most recent.
    classifier_corrections_hidden_layers: int = -1  # Wjets correction model hidden layers; -1 means reuse `classifier_hidden_layers`.
    plot_training_diagnostics: bool = True   # Plot training loss / learning-rate / time-per-epoch curves.
    plot_nf_sampling: bool = True            # Plot NF-sampled vs data histograms in training variables.
    plot_ff_results: bool = True             # Plot fake-factor comparison stacks for each njets category.
    plot_ar_data_with_clipping: bool = False  # Plot AR data with both kept and excluded events (by clipping mask).
    plot_taylor_coefficients: bool = True   # Compute and plot first-order Taylor coefficients (mean |d log p/d x_i|). Slow — needs a backward pass.
    plot_complete_variables: bool = False
    ratio_ylim_min: float = 0.75  # Lower y-limit for ratio panels.
    ratio_ylim_max: float = 1.25  # Upper y-limit for ratio panels.
# ------------ functions ----------

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
    return f"vars{len(variables)}_{readable_tail}_{variables_hash}"


def _build_training_variables_prefix(variables: list[str]) -> str:
    """Return the hash-free readable prefix, e.g. 'vars5_pt_vis'."""
    tail_variables = variables[4:]
    if tail_variables:
        readable_tail = "_".join(tail_variables)
        readable_tail = re.sub(r"[^A-Za-z0-9_]+", "_", readable_tail).strip("_")
    else:
        readable_tail = "none"
    return f"vars{len(variables)}_{readable_tail}"


def resolve_training_tag(variables: list[str], mode_dir: str, base_dir: str = 'Training_results_new') -> str:
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


def load_binary_classifier_checkpoint(checkpoint_dir: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_dir) / 'model_checkpoint.pth'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Binary classifier checkpoint not found: {checkpoint_path}')

    ckpt = torch.load(checkpoint_path, map_location=device)
    training_cfg = ckpt.get('training_config', {})
    feature_columns = ckpt.get('feature_columns', ckpt.get('variables', list(variables)))

    hidden_dim = int(training_cfg.get('hidden_dim', 200))
    hidden_layers = int(training_cfg.get('hidden_layers', 2))
    dropout = float(training_cfg.get('dropout', 0.15))

    model = BinaryClassifier(
        input_dim=len(feature_columns),
        hidden_dim=hidden_dim,
        p=dropout,
        hidden_layers=hidden_layers,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    prior_ar_over_sr = float(ckpt.get('prior_ar_over_sr', 1.0))
    return model, list(feature_columns), prior_ar_over_sr


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

def normalizing_flow_ff(
    df,
    variables,
    model_AR_like_wjets,
    model_SR_like_wjets,
    global_ff_wjets,
    model_AR_like_qcd,
    model_SR_like_qcd,
    global_ff_qcd,
    device,
    plotting=True,
    plot_dir="plots",
    include_njets=True,
    ff_estimator: str = 'nf',
    prior_ar_over_sr_wjets: float | None = None,
    prior_ar_over_sr_qcd: float | None = None,
    classifier_features_wjets: list[str] | None = None,
    classifier_features_qcd: list[str] | None = None,
    correction_model_wjets_dr=None,
    correction_model_wjets_antidr=None,
    correction_prior_ar_over_sr_wjets_dr: float | None = None,
    correction_prior_ar_over_sr_wjets_antidr: float | None = None,
    correction_features_wjets_dr: list[str] | None = None,
    correction_features_wjets_antidr: list[str] | None = None,
    correction_global_ff_wjets_dr: float | None = None,
    correction_global_ff_wjets_antidr: float | None = None,
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
    df = df.copy()
    if df.empty:
        df['ff_nf_wjets'] = pd.Series(dtype=float)
        df['ff_nf_qcd'] = pd.Series(dtype=float)
        df['ff_nf'] = pd.Series(dtype=float)
        return df

    input_variables = (['njets'] + variables) if include_njets else list(variables)
    ff_full_wjets_uncorrected = None
    clip_mask_wjets_uncorrected = None

    if ff_estimator == 'binary_classifier':
        wjets_features = classifier_features_wjets if classifier_features_wjets is not None else list(variables)
        qcd_features = classifier_features_qcd if classifier_features_qcd is not None else list(variables)
        prior_ar_over_sr_wjets = 1.0 if prior_ar_over_sr_wjets is None else float(prior_ar_over_sr_wjets)
        prior_ar_over_sr_qcd = 1.0 if prior_ar_over_sr_qcd is None else float(prior_ar_over_sr_qcd)

        df_pt_wjets = get_my_data_wjets(df, wjets_features).to_torch().to(device)
        ratio_wjets = evaluate_density_ratio_binary_classifier(
            model_AR_like_wjets,
            df_pt_wjets.X,
            prior_ar_over_sr=prior_ar_over_sr_wjets,
        )
        ff_full_wjets, _, global_ff_cor_wjets, clip_mask_wjets, clip_value_wjets = compute_eventwise_fake_factors_binary_classifier(
            ratio_wjets,
            global_ff_wjets,
        )
        ff_full_wjets_uncorrected = ff_full_wjets.copy()
        clip_mask_wjets_uncorrected = clip_mask_wjets.copy()

        if (
            correction_model_wjets_dr is not None
            and correction_model_wjets_antidr is not None
            and correction_global_ff_wjets_dr is not None
            and correction_global_ff_wjets_antidr is not None
            and correction_global_ff_wjets_dr > 0
            and correction_global_ff_wjets_antidr > 0
        ):
            corr_dr_features = correction_features_wjets_dr if correction_features_wjets_dr is not None else wjets_features
            corr_antidr_features = correction_features_wjets_antidr if correction_features_wjets_antidr is not None else wjets_features
            corr_prior_dr = 1.0 if correction_prior_ar_over_sr_wjets_dr is None else float(correction_prior_ar_over_sr_wjets_dr)
            corr_prior_antidr = 1.0 if correction_prior_ar_over_sr_wjets_antidr is None else float(correction_prior_ar_over_sr_wjets_antidr)

            df_pt_wjets_corr_dr = get_my_data_wjets(df, corr_dr_features).to_torch().to(device)
            ratio_wjets_corr_dr = evaluate_density_ratio_binary_classifier(
                correction_model_wjets_dr,
                df_pt_wjets_corr_dr.X,
                prior_ar_over_sr=corr_prior_dr,
            )
            ff_wjets_corr_dr, _, _, _, _ = compute_eventwise_fake_factors_binary_classifier(
                ratio_wjets_corr_dr,
                float(correction_global_ff_wjets_dr),
            )

            df_pt_wjets_corr_antidr = get_my_data_wjets(df, corr_antidr_features).to_torch().to(device)
            ratio_wjets_corr_antidr = evaluate_density_ratio_binary_classifier(
                correction_model_wjets_antidr,
                df_pt_wjets_corr_antidr.X,
                prior_ar_over_sr=corr_prior_antidr,
            )
            ff_wjets_corr_antidr, _, _, _, _ = compute_eventwise_fake_factors_binary_classifier(
                ratio_wjets_corr_antidr,
                float(correction_global_ff_wjets_antidr),
            )

            with np.errstate(divide='ignore', invalid='ignore'):
                wjets_corr_ratio = ff_wjets_corr_antidr / ff_wjets_corr_dr
            valid_corr_ratio = np.isfinite(wjets_corr_ratio) & (ff_wjets_corr_dr > 0)
            wjets_corr_ratio = np.where(valid_corr_ratio, wjets_corr_ratio, 1.0)

            ff_full_wjets = ff_full_wjets * wjets_corr_ratio
            clip_mask_wjets = clip_mask_wjets & np.isfinite(ff_full_wjets) & (ff_full_wjets <= clip_value_wjets)

        df_pt_qcd = get_my_data_qcd(df, qcd_features).to_torch().to(device)
        ratio_qcd = evaluate_density_ratio_binary_classifier(
            model_AR_like_qcd,
            df_pt_qcd.X,
            prior_ar_over_sr=prior_ar_over_sr_qcd,
        )
        ff_full_qcd, _, global_ff_cor_qcd, clip_mask_qcd, clip_value_qcd = compute_eventwise_fake_factors_binary_classifier(
            ratio_qcd,
            global_ff_qcd,
        )
    else:
        # --- W+jets: evaluate PDFs on the full df ---
        df_pt_wjets = get_my_data_wjets(df, input_variables).to_torch().to(device)

        pdf_AR_like_wjets = evaluate_pdf(model_AR_like_wjets, df_pt_wjets.X)
        pdf_SR_like_wjets = evaluate_pdf(model_SR_like_wjets, df_pt_wjets.X)

        ff_full_wjets, _, global_ff_cor_wjets, clip_mask_wjets, clip_value_wjets = compute_eventwise_fake_factors(
            pdf_AR_like_wjets, pdf_SR_like_wjets, global_ff_wjets
        )

        # --- QCD FF ---
        # Evaluate QCD PDFs on the same full df (before any W+jets filtering)
        df_pt_qcd = get_my_data_qcd(df, input_variables).to_torch().to(device)

        pdf_AR_like_qcd = evaluate_pdf(model_AR_like_qcd, df_pt_qcd.X)
        pdf_SR_like_qcd = evaluate_pdf(model_SR_like_qcd, df_pt_qcd.X)

        ff_full_qcd, _, global_ff_cor_qcd, clip_mask_qcd, clip_value_qcd = compute_eventwise_fake_factors(
            pdf_AR_like_qcd, pdf_SR_like_qcd, global_ff_qcd
        )

    # Keep per-process clipping/correction independent.
    # `compute_eventwise_fake_factors` already applies each process-specific
    # global correction. A second combined correction can strongly over-scale FFs.
    combined_mask = clip_mask_wjets & clip_mask_qcd

    if not np.any(combined_mask):
        logger.warning("No events survive joint FF clipping; returning empty dataframe.")
        df = df.iloc[0:0].copy()
        df['ff_nf_wjets'] = pd.Series(dtype=float)
        df['ff_nf_qcd'] = pd.Series(dtype=float)
        df['ff_nf'] = pd.Series(dtype=float)
        return df

    logger.info(
        "%s clipping acceptance: Wjets=%.4f, QCD=%.4f, joint=%.4f",
        'Classifier' if ff_estimator == 'binary_classifier' else 'NF',
        float(np.mean(clip_mask_wjets)),
        float(np.mean(clip_mask_qcd)),
        float(np.mean(combined_mask)),
    )

    wjets_clipped_percent = 100.0 * (1.0 - float(np.mean(clip_mask_wjets)))
    qcd_clipped_percent = 100.0 * (1.0 - float(np.mean(clip_mask_qcd)))
    joint_clipped_percent = 100.0 * (1.0 - float(np.mean(combined_mask)))

    df = df[combined_mask].copy()
    df['ff_nf_wjets'] = ff_full_wjets[combined_mask]
    df['ff_nf_qcd'] = ff_full_qcd[combined_mask]

    # --- plotting ---
    if plotting:
        import matplotlib.pyplot as plt
        ff_clip_wjets = clip_value_wjets
        ff_clip_qcd = clip_value_qcd

        bins = np.logspace(-3, 1, 61)
        ff_kept_wjets = ff_full_wjets[clip_mask_wjets]
        ff_clipped_wjets = ff_full_wjets[~clip_mask_wjets]
        ff_kept_qcd = ff_full_qcd[clip_mask_qcd]
        ff_clipped_qcd = ff_full_qcd[~clip_mask_qcd]

        fig, ax = plt.subplots(2, 1, figsize=(8, 7))

        if ff_full_wjets_uncorrected is not None and clip_mask_wjets_uncorrected is not None:
            ff_kept_wjets_uncorrected = ff_full_wjets_uncorrected[clip_mask_wjets_uncorrected]
            ax[0].hist(ff_kept_wjets_uncorrected, bins=bins, label="W+jets FF (no correction)", color="#e76300", alpha=0.45)
            ax[0].hist(ff_kept_wjets, bins=bins, label="W+jets FF (with correction)", color="#e76300", alpha=0.9)
            ax[0].hist(ff_clipped_wjets, bins=bins, label="W+jets FF clipped (with correction)", color="#e76300", alpha=0.18)
        else:
            ax[0].hist(ff_kept_wjets, bins=bins, label="W+jets FF (kept)", color="#e76300", alpha=0.9)
            ax[0].hist(ff_clipped_wjets, bins=bins, label="W+jets FF (clipped)", color="#e76300", alpha=0.25)
        ax[0].axvline(ff_clip_wjets, color="black", linestyle="--", linewidth=1.4, label=fr"W+jets clip ({ff_clip_wjets:.2f})")
        ax[0].set_xscale("log")
        ax[0].set_yscale('log')
        ax[0].set_xlim(1e-3, 1e1)
        ax[0].set_ylabel("Events")
        ax[0].set_title("W+jets eventwise FF", pad=30)
        ax[0].text(
            0.98,
            0.94,
            f"Clipped: {wjets_clipped_percent:.2f}%",
            transform=ax[0].transAxes,
            ha='right',
            va='top',
            fontsize=10,
        )
        CMS_CHANNEL_TITLE([ax[0]])
        CMS_LUMI_TITLE([ax[0]])
        CMS_LABEL([ax[0]])
        CMS_NJETS_TITLE([ax[0]], title=r"$N_{jets} \geq 0$")

        ax[1].hist(ff_kept_qcd, bins=bins, label="QCD FF (kept)", color="#b9ac70", alpha=0.9)
        ax[1].hist(ff_clipped_qcd, bins=bins, label="QCD FF (clipped)", color="#b9ac70", alpha=0.25)
        ax[1].axvline(ff_clip_qcd, color="black", linestyle="--", linewidth=1.4, label=fr"QCD clip ({ff_clip_qcd:.2f})")
        ax[1].set_xscale("log")
        ax[1].set_yscale('log')
        ax[1].set_xlim(1e-3, 1e1)
        ax[1].set_ylabel("Events")
        ax[1].set_xlabel("Eventwise FF")
        ax[1].text(
            0.98,
            0.94,
            f"Clipped: {qcd_clipped_percent:.2f}%",
            transform=ax[1].transAxes,
            ha='right',
            va='top',
            fontsize=10,
        )

        handles0, labels0 = ax[0].get_legend_handles_labels()
        handles1, labels1 = ax[1].get_legend_handles_labels()
        fig.legend(handles0 + handles1, labels0 + labels1,
                   loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=3, frameon=False, fontsize=9)

        #fig.text(0.5, 0.955, f"Joint clipped (W+jets & QCD): {joint_clipped_percent:.2f}%", ha='center', va='center', fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_dir / 'hist_FF.png')
        plt.close(fig)

    # --- assemble combined NF fake factor (process-fraction weighted) ---
    _ff_file = cr.CorrectionSet.from_file('/work/mmoser/TauFakeFactors/workdir/ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz')
    _corr_file = cr.CorrectionSet.from_file('/work/mmoser/TauFakeFactors/workdir/ff_2026_01_19_check_variable/2018/FF_corrections_et.json.gz')

    _frac = _ff_file['process_fractions']
    _ff_ttbar = _ff_file['ttbar_fake_factors']

    df['ttbar_classic_ff'] = _ff_ttbar.evaluate(
        df.pt_2.values,
        df.njets.values,
        "nominal",
    )
    df['ttbar_corrected_classic_ff'] = df['ttbar_classic_ff'] * _corr_file.compound["ttbar_compound_correction"].evaluate(
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

    df['process_fraction_wjets'] = _frac.evaluate('Wjets', df.mt_1.values, df.njets.values, 'nominal')
    df['process_fraction_qcd'] = _frac.evaluate('QCD', df.mt_1.values, df.njets.values, 'nominal')
    df['process_fraction_ttbar'] = _frac.evaluate('ttbar', df.mt_1.values, df.njets.values, 'nominal')

    df['ff_nf'] = (
        df['process_fraction_wjets'] * df['ff_nf_wjets']
        + df['process_fraction_qcd'] * df['ff_nf_qcd']
        + df['process_fraction_ttbar'] * df['ttbar_corrected_classic_ff']
    )

    return df, combined_mask


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

def mask_DR_wjets(df):                  # without SS/OS conditions !!!!!!!!!!!!11

    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 > 70)
    mask_DR = (mask_a1 & mask_a2  & mask_a4 & mask_a5 & mask_a6)

    return df[mask_DR].copy()


def mask_antiDR_wjets(df):              # without SS/OS conditions !!!!!!!!!!!!11

    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 <= 70)
    mask_anti_dr = (mask_a1 & mask_a2 & mask_a4 & mask_a5 & mask_a6)

    return df[mask_anti_dr].copy()

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

def mask_preselection_tight(df):
    mask_eta = (df.eta_1 <= 2.1) & (df.eta_2 <= 2.3)
    mask_pt = (df.pt_1 >= 35) & (df.pt_2 >= 32)
    #mask_m_vis = (df.m_vis >= 35)
    mask_tau_decay_mode = (df.tau_decaymode_2 == 0) | (df.tau_decaymode_2 == 1) | (df.tau_decaymode_2 == 10) | (df.tau_decaymode_2 == 11)
    return df[mask_eta & mask_pt &  mask_tau_decay_mode]


def mask_preselection_tight_binary_classifier(df):
    mask_eta = (df.eta_1 <= 2.1) & (df.eta_2 <= 2.3)
    mask_pt = (df.pt_1 >= 33) & (df.pt_2 >= 30)
    mask_tau_decay_mode = (df.tau_decaymode_2 == 0) | (df.tau_decaymode_2 == 1) | (df.tau_decaymode_2 == 10) | (df.tau_decaymode_2 == 11)
    return df[mask_eta & mask_pt & mask_tau_decay_mode]


def mask_preselection_for_estimator(df):
    if args.ff_estimator == 'binary_classifier':
        return mask_preselection_tight_binary_classifier(df)
    return mask_preselection_tight(df)

def SR(df):                 # without SS/OS conditions !!!!!!!!!!!!11
    mask_s1 = (df.id_tau_vsJet_Tight_2 > 0.5)
    mask_s2 = (df.nbtag == 0)
    mask_s3 = (df.q_1 * df.q_2 < 0)
    mask_s4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_s5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_s6 = (df.mt_1 < 70)

    mask_SR_like = (mask_s1 & mask_s4 & mask_s5 & mask_s6)

    return(df[mask_SR_like])

def AR(df):                 # without SS/OS conditions !!!!!!!!!!!!11
    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5) & (df.id_tau_vsJet_Tight_2 < 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 < 70)

    mask_AR_like = (mask_a1 &  mask_a4 & mask_a5 & mask_a6)

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

args = Args(explicit_bool=True).parse_args()

dim = len(variables)
training_variables_tag = build_training_variables_tag(variables)

variables_with_njets = ['njets'] + variables

# --------------- device -------------

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ----------------load model -------------------

config_path = "../configs/config_NF.yaml"

_MODE_DIR = {
    'grouped_njets_split': 'split_njets_0_1_ge2',
    'single_nf':           'no_njets_split',
    'conditional_nf':      'conditional_njets_input',
}
mode_dir = _MODE_DIR[args.model_mode]
include_njets_feature = args.model_mode in ('grouped_njets_split', 'conditional_nf')

classifier_features_wjets = None
classifier_features_qcd = None
prior_ar_over_sr_wjets = None
prior_ar_over_sr_qcd = None
correction_model_wjets_dr = None
correction_model_wjets_antidr = None
correction_features_wjets_dr = None
correction_features_wjets_antidr = None
correction_prior_ar_over_sr_wjets_dr = None
correction_prior_ar_over_sr_wjets_antidr = None

if args.ff_estimator == 'binary_classifier':
    include_njets_feature = False
    classifier_base = Path('Training_results_new') / 'binary_classifier'
    classifier_prefix = f'training_hl{args.classifier_hidden_layers}_'
    logger.info('Searching classifier trainings with prefix: %s', classifier_prefix)
    resolved_tag = args.classifier_training_tag or resolve_latest_training_tag(str(classifier_base), prefix=classifier_prefix)
    logger.info('Using binary-classifier training tag: %s', resolved_tag)

    chk_pth_model_AR_like_wjets = f'Training_results_new/binary_classifier/training_{resolved_tag}/Wjets/all/SR_AR_classifier/latest'
    chk_pth_model_SR_like_wjets = chk_pth_model_AR_like_wjets
    chk_pth_model_AR_like_qcd = f'Training_results_new/binary_classifier/training_{resolved_tag}/QCD/all/SR_AR_classifier/latest'
    chk_pth_model_SR_like_qcd = chk_pth_model_AR_like_qcd

    model_AR_like_wjets, classifier_features_wjets, prior_ar_over_sr_wjets = load_binary_classifier_checkpoint(
        chk_pth_model_AR_like_wjets,
        device=device,
    )
    model_SR_like_wjets = model_AR_like_wjets
    model_AR_like_qcd, classifier_features_qcd, prior_ar_over_sr_qcd = load_binary_classifier_checkpoint(
        chk_pth_model_AR_like_qcd,
        device=device,
    )
    model_SR_like_qcd = model_AR_like_qcd

    if args.apply_wjets_binary_correction:
        correction_base = Path('Training_results_new') / 'binary_classifier_corrections'
        correction_hidden_layers = args.classifier_hidden_layers if args.classifier_corrections_hidden_layers < 0 else args.classifier_corrections_hidden_layers
        correction_prefix = f'training_hl{correction_hidden_layers}_'
        logger.info('Searching correction trainings with prefix: %s', correction_prefix)
        resolved_correction_tag = args.classifier_corrections_training_tag or resolve_latest_training_tag(
            str(correction_base),
            prefix=correction_prefix,
        )
        logger.info('Using binary-classifier correction training tag: %s', resolved_correction_tag)

        chk_pth_corr_wjets_dr, chk_pth_corr_wjets_antidr = correction_classifier_paths(resolved_correction_tag)
        correction_model_wjets_dr, correction_features_wjets_dr, correction_prior_ar_over_sr_wjets_dr = load_binary_classifier_checkpoint(
            chk_pth_corr_wjets_dr,
            device=device,
        )
        correction_model_wjets_antidr, correction_features_wjets_antidr, correction_prior_ar_over_sr_wjets_antidr = load_binary_classifier_checkpoint(
            chk_pth_corr_wjets_antidr,
            device=device,
        )
    else:
        logger.info('Wjets binary antiDR/DR correction disabled via args.apply_wjets_binary_correction=False')

else:
    # Resolve the training folder by glob (ignores hash mismatches in the suffix).
    resolved_tag = resolve_training_tag(variables, mode_dir)
    logger.info('Using training tag: %s (exact computed: %s)', resolved_tag, training_variables_tag)

    chk_pth_model_AR_like_wjets = f'Training_results_new/{mode_dir}/training_{resolved_tag}/Wjets/all/AR-like/latest'
    chk_pth_model_SR_like_wjets = f'Training_results_new/{mode_dir}/training_{resolved_tag}/Wjets/all/SR-like/latest'

    chk_pth_model_AR_like_qcd = f'Training_results_new/{mode_dir}/training_{resolved_tag}/QCD/all/AR-like/latest'
    chk_pth_model_SR_like_qcd = f'Training_results_new/{mode_dir}/training_{resolved_tag}/QCD/all/SR-like/latest'

    config_AR_like_wjets = load_saved_model_config(chk_pth_model_AR_like_wjets, config_path)
    config_SR_like_wjets = load_saved_model_config(chk_pth_model_SR_like_wjets, config_path)
    config_AR_like_qcd = load_saved_model_config(chk_pth_model_AR_like_qcd, config_path)
    config_SR_like_qcd = load_saved_model_config(chk_pth_model_SR_like_qcd, config_path)

    if args.model_mode == 'grouped_njets_split':
        model_AR_like_wjets = load_grouped_wjets_njets_router(
            checkpoint_dir=chk_pth_model_AR_like_wjets,
            config_path=config_path,
            variables=variables,
            device=device,
        )
        model_SR_like_wjets = load_grouped_wjets_njets_router(
            checkpoint_dir=chk_pth_model_SR_like_wjets,
            config_path=config_path,
            variables=variables,
            device=device,
        )
        model_AR_like_qcd = load_grouped_qcd_njets_router(
            checkpoint_dir=chk_pth_model_AR_like_qcd,
            config_path=config_path,
            variables=variables,
            device=device,
        )
        model_SR_like_qcd = load_grouped_qcd_njets_router(
            checkpoint_dir=chk_pth_model_SR_like_qcd,
            config_path=config_path,
            variables=variables,
            device=device,
        )
    elif args.model_mode == 'conditional_nf':
        model_AR_like_wjets = load_conditional_flow(dim=dim, cfg=config_AR_like_wjets, checkpoint_path=f'{chk_pth_model_AR_like_wjets}/model_checkpoint.pth', device=device)
        model_SR_like_wjets = load_conditional_flow(dim=dim, cfg=config_SR_like_wjets, checkpoint_path=f'{chk_pth_model_SR_like_wjets}/model_checkpoint.pth', device=device)
        model_AR_like_qcd   = load_conditional_flow(dim=dim, cfg=config_AR_like_qcd, checkpoint_path=f'{chk_pth_model_AR_like_qcd}/model_checkpoint.pth',   device=device)
        model_SR_like_qcd   = load_conditional_flow(dim=dim, cfg=config_SR_like_qcd, checkpoint_path=f'{chk_pth_model_SR_like_qcd}/model_checkpoint.pth',   device=device)
    else:  # single_nf
        model_AR_like_wjets = load_flow(dim=dim, cfg=config_AR_like_wjets, checkpoint_path=f'{chk_pth_model_AR_like_wjets}/model_checkpoint.pth', device=device)
        model_SR_like_wjets = load_flow(dim=dim, cfg=config_SR_like_wjets, checkpoint_path=f'{chk_pth_model_SR_like_wjets}/model_checkpoint.pth', device=device)
        model_AR_like_qcd   = load_flow(dim=dim, cfg=config_AR_like_qcd, checkpoint_path=f'{chk_pth_model_AR_like_qcd}/model_checkpoint.pth',   device=device)
        model_SR_like_qcd   = load_flow(dim=dim, cfg=config_SR_like_qcd, checkpoint_path=f'{chk_pth_model_SR_like_qcd}/model_checkpoint.pth',   device=device)


# ----- load data -----

data_complete = pd.read_feather('../../data/data_complete.feather')


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

if args.plot_complete_variables == True:
    list_variables = [
        "pt_1","pt_2","eta_1","eta_2","jpt_1","jpt_2","jeta_1","jeta_2",
        "m_fastmtt","pt_fastmtt","met","njets","mt_tot","m_vis",
        "pt_tt","pt_vis","mjj","pt_dijet","pt_ttjj","deltaEta_jj","deltaR_jj",
        "deltaR_ditaupair","deltaR_1j1","deltaR_1j2", "nbtag", "mt_1", "mt_2", "iso_1", "iso_2",
        "deltaR_2j1","deltaR_2j2","deltaR_12j1","deltaR_12j2","deltaEta_1j1",
        "deltaEta_1j2","deltaEta_2j1","deltaEta_2j2","deltaEta_12j1","deltaEta_12j2", 'tau_decaymode_1', 'tau_decaymode_2',
        "mass_1", "mass_2",
]

else:
    list_variables = ['pt_1', 'pt_2', 'm_vis', 'deltaR_ditaupair', 'pt_vis', 'pt_tt','m_fastmtt', 
                      'eta_1', 'eta_2', 'met', 'mt_1', 'mt_2']
with open('../configs/labels.yaml', 'r') as f:
    labels = yaml.safe_load(f)
labels = labels['et']

# Load short labels for Taylor coefficient plots
with open('../configs/labels_short.yaml', 'r') as f:
    labels_short = yaml.safe_load(f)
labels_short = labels_short['et']
list_xlabels = [labels[k] for k in list_variables]

bins_by_variable = {
    # Existing defaults (kept identical to plot_complete_variables == False setup)
    'pt_1': np.linspace(0, 150, 31),
    'pt_2': np.linspace(0, 150, 31),
    'm_vis': np.linspace(0, 220, 31),
    'deltaR_ditaupair': np.linspace(0, 5, 21),
    'pt_vis': np.linspace(0, 160, 31),
    'pt_tt': np.linspace(0, 160, 31),
    'm_fastmtt': np.linspace(0, 220, 31),
    'eta_1': np.linspace(-3, 3, 31),
    'eta_2': np.linspace(-3, 3, 31),
    'met': np.linspace(0, 150, 31),
    'mt_1': np.linspace(0, 150, 31),
    'mt_2': np.linspace(0, 150, 31),

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
}

list_bins = [np.asarray(bins_by_variable[var]) for var in list_variables]
main_plot_bins_by_variable = {var: np.asarray(bins_by_variable[var]) for var in list_variables}


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


sampling_plot_bins_by_variable = _build_sampling_bins_from_main(main_plot_bins_by_variable, max_scale=2.0)

if args.ff_estimator == 'binary_classifier':
    correction_mode_dir = 'with_corrections' if args.apply_wjets_binary_correction else 'without_corrections'
    plot_root_dir = Path('plots') / 'binary_classifier' / correction_mode_dir / f"training_{resolved_tag}"
else:
    plot_root_dir = Path('plots') / mode_dir / f"training_{resolved_tag}"
plot_root_dir.mkdir(parents=True, exist_ok=True)

logger.info("Plot output root: %s", plot_root_dir)

training_log_specs = [
    ('Wjets AR-like', chk_pth_model_AR_like_wjets),
    ('Wjets SR-like', chk_pth_model_SR_like_wjets),
    ('QCD AR-like', chk_pth_model_AR_like_qcd),
    ('QCD SR-like', chk_pth_model_SR_like_qcd),
]
if args.ff_estimator == 'binary_classifier':
    training_log_specs = [
        ('Wjets SR/AR classifier', chk_pth_model_AR_like_wjets),
        ('QCD SR/AR classifier', chk_pth_model_AR_like_qcd),
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

    wjets_ar_data = AR_like_wjets(data_preselected)
    wjets_ar_data = wjets_ar_data[(wjets_ar_data.process == 0) & (wjets_ar_data.OS == True)].copy()
    wjets_sr_data = SR_like_wjets(data_preselected)
    wjets_sr_data = wjets_sr_data[(wjets_sr_data.process == 0) & (wjets_sr_data.OS == True)].copy()

    qcd_ar_data = AR_like_qcd(data_preselected)
    qcd_ar_data = qcd_ar_data[(qcd_ar_data.process == 0) & (qcd_ar_data.SS == True)].copy()
    qcd_sr_data = SR_like_qcd(data_preselected)
    qcd_sr_data = qcd_sr_data[(qcd_sr_data.process == 0) & (qcd_sr_data.SS == True)].copy()

    panel_specs = [
        ('Wjets AR-like', wjets_ar_data, model_AR_like_wjets, '#1f77b4'),
        ('Wjets SR-like', wjets_sr_data, model_SR_like_wjets, '#17becf'),
        ('QCD AR-like', qcd_ar_data, model_AR_like_qcd, '#d62728'),
        ('QCD SR-like', qcd_sr_data, model_SR_like_qcd, '#ff7f0e'),
    ]

    n_samples = 10_000
    for var in variables:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False, sharey=False)
        flat_axes = axes.flatten()
        fixed_bins = sampling_plot_bins_by_variable.get(var)

        for axis, (title, data_df, model, color) in zip(flat_axes, panel_specs):
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

            axis.hist(
                sampled_values,
                bins=bins,
                density=True,
                histtype='stepfilled',
                alpha=0.35,
                color=color,
                label=f'NF sampled ({n_samples})',
            )
            axis.hist(
                data_values,
                bins=bins,
                density=True,
                histtype='step',
                linewidth=1.8,
                color='black',
                label=f'Data ({len(data_values)})',
            )

            axis.set_title(title)
            axis.set_xlabel(labels.get(var, var))
            axis.set_ylabel('Density')
            axis.set_yscale('log')
            axis.grid(True, linestyle=':', alpha=0.35)
            axis.legend(frameon=False, loc='best')
            axis.set_xlim(float(bins[0]), float(bins[-1]))

        fig.suptitle(
            f'NF sampled vs data ({category_name}, {njets_title})\nTraining variable: {labels.get(var, var)}',
            fontsize=16,
            y=1.01,
        )
        fig.tight_layout()
        fig.savefig(sampling_plot_dir / f'nf_sample_vs_data_{var}.png', bbox_inches='tight')
        fig.savefig(sampling_plot_dir / f'nf_sample_vs_data_{var}.pdf', bbox_inches='tight')
        plt.close(fig)

    logger.info('Saved NF sampling validation plots to %s', sampling_plot_dir)

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
    from tayloranalysis import extend_model as _ta_extend

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


def _add_cms_privatework_lumi_row(axis, y: float = 1.005, fontsize: int = 9) -> None:
    axis.text(
        0.0,
        y,
        'Private work (CMS data/simulation)',
        ha='left',
        va='bottom',
        fontsize=fontsize,
        fontproperties='Tex Gyre Heros:italic',
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.text(
        1.0,
        y,
        r'59.8 $fb^{-1}$ (2018, 13 TeV)',
        ha='right',
        va='bottom',
        fontsize=fontsize,
        fontproperties='Tex Gyre Heros',
        transform=axis.transAxes,
        clip_on=False,
    )


def plot_nf_taylor_analysis(output_dir: Path) -> None:
    """
    Compute and plot first-order Taylor coefficients for all four NF models.
    Produces a 2x2 figure with horizontal bar charts sorted by |TC| magnitude.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data_pre = mask_preselection_for_estimator(data_complete)
    wjets_ar = AR_like_wjets(data_pre)
    wjets_ar = wjets_ar[(wjets_ar.process == 0) & (wjets_ar.OS == True)].copy()
    wjets_sr = SR_like_wjets(data_pre)
    wjets_sr = wjets_sr[(wjets_sr.process == 0) & (wjets_sr.OS == True)].copy()
    qcd_ar = AR_like_qcd(data_pre)
    qcd_ar = qcd_ar[(qcd_ar.process == 0) & (qcd_ar.SS == True)].copy()
    qcd_sr = SR_like_qcd(data_pre)
    qcd_sr = qcd_sr[(qcd_sr.process == 0) & (qcd_sr.SS == True)].copy()

    panel_specs = [
        ('Wjets AR-like', wjets_ar, model_AR_like_wjets, '#1f77b4'),
        ('Wjets SR-like', wjets_sr, model_SR_like_wjets, '#17becf'),
        ('QCD AR-like',   qcd_ar,   model_AR_like_qcd,   '#d62728'),
        ('QCD SR-like',   qcd_sr,   model_SR_like_qcd,   '#ff7f0e'),
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
            _add_cms_privatework_lumi_row(axis)

            single_fig, single_axis = plt.subplots(figsize=(9.5, 7.2))
            single_axis.text(0.5, 0.5, f'Failed:\n{exc}', ha='center', va='center',
                             transform=single_axis.transAxes, fontsize=10)
            single_axis.set_title(title)
            _add_cms_privatework_lumi_row(single_axis, fontsize=10)
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
            _add_cms_privatework_lumi_row(axis)

            single_fig, single_axis = plt.subplots(figsize=(9.5, 7.2))
            single_axis.text(0.5, 0.5, 'No data', ha='center', va='center', transform=single_axis.transAxes)
            single_axis.set_title(title)
            _add_cms_privatework_lumi_row(single_axis, fontsize=10)
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
        _add_cms_privatework_lumi_row(axis)
        axis.grid(True, axis='x', linestyle=':', alpha=0.4)
        axis.tick_params(direction='in')

        single_fig, single_axis = plt.subplots(figsize=(6, 5))
        single_axis.barh(y_pos, tc_values, color=color, alpha=0.75, edgecolor='none')
        single_axis.set_yticks(y_pos)
        single_axis.set_yticklabels(display_names, fontsize=10)
        single_axis.set_xlabel(r"$\tilde{c_i}$")
        #single_axis.set_xlabel(r'$\sigma_i\,\langle\,|\,\partial_i \log p(x)\,|\,\rangle$')
        #single_axis.set_title(title)
        _add_cms_privatework_lumi_row(single_axis, fontsize=10)
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


def plot_nf_second_order_covariance(output_dir: Path) -> None:
    """
    Compute and plot second-order Taylor coefficient matrices for all four NF models.
    Each model produces a heatmap of mean |d² log p / dx_i dx_j|.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data_pre = mask_preselection_for_estimator(data_complete)
    wjets_ar = AR_like_wjets(data_pre)
    wjets_ar = wjets_ar[(wjets_ar.process == 0) & (wjets_ar.OS == True)].copy()
    wjets_sr = SR_like_wjets(data_pre)
    wjets_sr = wjets_sr[(wjets_sr.process == 0) & (wjets_sr.OS == True)].copy()
    qcd_ar = AR_like_qcd(data_pre)
    qcd_ar = qcd_ar[(qcd_ar.process == 0) & (qcd_ar.SS == True)].copy()
    qcd_sr = SR_like_qcd(data_pre)
    qcd_sr = qcd_sr[(qcd_sr.process == 0) & (qcd_sr.SS == True)].copy()

    panel_specs = [
        ('Wjets AR-like', wjets_ar, model_AR_like_wjets),
        ('Wjets SR-like', wjets_sr, model_SR_like_wjets),
        ('QCD AR-like',   qcd_ar,   model_AR_like_qcd),
        ('QCD SR-like',   qcd_sr,   model_SR_like_qcd),
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
            _add_cms_privatework_lumi_row(axis)

            single_fig, single_axis = plt.subplots(figsize=(9.5, 7.8))
            single_axis.text(0.5, 0.5, f'Failed:\n{exc}', ha='center', va='center',
                             transform=single_axis.transAxes, fontsize=10)
            single_axis.set_title(title)
            _add_cms_privatework_lumi_row(single_axis, fontsize=10)
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
            _add_cms_privatework_lumi_row(axis)

            single_fig, single_axis = plt.subplots(figsize=(9.5, 7.8))
            single_axis.text(0.5, 0.5, 'No data', ha='center', va='center', transform=single_axis.transAxes)
            single_axis.set_title(title)
            _add_cms_privatework_lumi_row(single_axis, fontsize=10)
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
        _add_cms_privatework_lumi_row(axis)

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
        _add_cms_privatework_lumi_row(single_axis, fontsize=10)
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

    ax_main.set_ylabel("Events")
    ax_main.legend(loc='upper right', frameon=False, fontsize=12)
    ax_main.tick_params(direction='in', top=True, right=True)

    ax_ratio.stairs(excluded_fraction, bins, color='black', linewidth=1.4)
    ax_ratio.axhline(0.0, color='gray', linestyle=':', linewidth=1.0)
    ax_ratio.set_ylim(0.0, 1.0)
    ax_ratio.set_ylabel("Excluded / Total")
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.grid(True, linestyle=':', alpha=0.6)
    ax_ratio.tick_params(direction='in', top=True, right=True)
    
    fig.savefig(output_dir / f'{var}_ar_clipping.png', dpi=150)
    fig.savefig(output_dir / f'{var}_ar_clipping.pdf')
    plt.close(fig)


def run_plots_for_njets_category(category_name, njets_title):
    category_plot_dir = plot_root_dir / category_name
    category_plot_dir.mkdir(parents=True, exist_ok=True)

    data_complete_njets = select_njets_category(data_complete, category_name)
    data_preselected = mask_preselection_for_estimator(data_complete_njets)

    if args.plot_nf_sampling and args.ff_estimator == 'nf':
        plot_nf_sampling_training_variables(category_name, njets_title, data_preselected)
    if not args.plot_ff_results:
        return

    logger.info(
        "Starting %s: %d input events, %d after preselection",
        category_name,
        len(data_complete_njets),
        len(data_preselected),
    )

    data_AR = AR(data_preselected)
    data_AR = data_AR[data_AR.OS == True]
    data_SR = SR(data_preselected)

    data_AR_OS = data_AR[(data_AR.process == 0)].copy()
    data_AR_like_wjets = AR_like_wjets(data_preselected)
    data_SR_like_wjets = SR_like_wjets(data_preselected)
    data_AR_like_qcd = AR_like_qcd(data_preselected)
    data_SR_like_qcd = SR_like_qcd(data_preselected)

    data_AR_like_OS_wjets = data_AR_like_wjets[(data_AR_like_wjets.process == 0) & (data_AR_like_wjets.OS == True)]
    data_SR_like_OS_wjets = data_SR_like_wjets[(data_SR_like_wjets.process == 0) & (data_SR_like_wjets.OS == True)]

    data_AR_like_SS_qcd = data_AR_like_qcd[(data_AR_like_qcd.process == 0) & (data_AR_like_qcd.SS == True)]
    data_SR_like_SS_qcd = data_SR_like_qcd[(data_SR_like_qcd.process == 0) & (data_SR_like_qcd.SS == True)]

    required_samples = {
        'AR_like_OS_wjets': data_AR_like_OS_wjets,
        'SR_like_OS_wjets': data_SR_like_OS_wjets,
        'AR_like_SS_qcd': data_AR_like_SS_qcd,
        'SR_like_SS_qcd': data_SR_like_SS_qcd,
    }
    empty_required = [name for name, sample in required_samples.items() if sample.empty]
    if empty_required:
        logger.warning('Skipping %s because required samples are empty: %s', category_name, ', '.join(empty_required))
        return

    data_SR_OS = data_SR[(data_SR.OS == True)]

    global_ff_wjets = len(data_SR_like_OS_wjets) / len(data_AR_like_OS_wjets)
    global_ff_qcd = len(data_SR_like_SS_qcd) / len(data_AR_like_SS_qcd)

    global_ff_wjets_dr_correction = global_ff_wjets
    global_ff_wjets_antidr_correction = None
    if args.ff_estimator == 'binary_classifier' and args.apply_wjets_binary_correction and correction_model_wjets_dr is not None and correction_model_wjets_antidr is not None:
        data_antidr_wjets = mask_antiDR_wjets(data_preselected)
        data_AR_like_OS_wjets_antidr = data_antidr_wjets[
            (data_antidr_wjets.id_tau_vsJet_VLoose_2 > 0.5)
            & (data_antidr_wjets.id_tau_vsJet_Tight_2 < 0.5)
            & (data_antidr_wjets.process == 0)
            & (data_antidr_wjets.OS == True)
        ]
        data_SR_like_OS_wjets_antidr = data_antidr_wjets[
            (data_antidr_wjets.id_tau_vsJet_Tight_2 > 0.5)
            & (data_antidr_wjets.process == 0)
            & (data_antidr_wjets.OS == True)
        ]
        if len(data_AR_like_OS_wjets_antidr) > 0 and len(data_SR_like_OS_wjets_antidr) > 0:
            global_ff_wjets_antidr_correction = len(data_SR_like_OS_wjets_antidr) / len(data_AR_like_OS_wjets_antidr)
        else:
            logger.warning(
                'Skipping Wjets antiDR/DR FF correction in %s because antiDR SR/AR sample is empty.',
                category_name,
            )

    logger.info(
        "Prepared %s: AR(OS)=%d, SR(OS)=%d, Wjets FF=%.4f, QCD FF=%.4f",
        category_name,
        len(data_AR_OS),
        len(data_SR_OS[(data_SR_OS.process == 0)]),
        global_ff_wjets,
        global_ff_qcd,
    )

    data_AR_OS_nf, ar_os_clipping_mask = normalizing_flow_ff(
        data_AR_OS,
        variables,
        model_AR_like_wjets,
        model_SR_like_wjets,
        global_ff_wjets,
        model_AR_like_qcd,
        model_SR_like_qcd,
        global_ff_qcd,
        device,
        plotting=True,
        plot_dir=category_plot_dir,
        include_njets=include_njets_feature,
        ff_estimator=args.ff_estimator,
        prior_ar_over_sr_wjets=prior_ar_over_sr_wjets,
        prior_ar_over_sr_qcd=prior_ar_over_sr_qcd,
        classifier_features_wjets=classifier_features_wjets,
        classifier_features_qcd=classifier_features_qcd,
        correction_model_wjets_dr=correction_model_wjets_dr,
        correction_model_wjets_antidr=correction_model_wjets_antidr,
        correction_prior_ar_over_sr_wjets_dr=correction_prior_ar_over_sr_wjets_dr,
        correction_prior_ar_over_sr_wjets_antidr=correction_prior_ar_over_sr_wjets_antidr,
        correction_features_wjets_dr=correction_features_wjets_dr,
        correction_features_wjets_antidr=correction_features_wjets_antidr,
        correction_global_ff_wjets_dr=global_ff_wjets_dr_correction,
        correction_global_ff_wjets_antidr=global_ff_wjets_antidr_correction,
    )

    data_diboson_AR_OS = data_AR[((data_AR.process == 2) | (data_AR.process == 3))]
    data_DY_AR_OS = data_AR[(data_AR.process == 4) | (data_AR.process == 5)]
    data_ST_AR_OS = data_AR[(data_AR.process == 6) | (data_AR.process == 7)]
    data_ttbar_L_AR_OS = data_AR[(data_AR.process == 9)]
    data_embedding_AR_OS = data_AR[(data_AR.process == 10)]

    data_diboson_AR_OS_nf, _ = normalizing_flow_ff(data_diboson_AR_OS, variables, model_AR_like_wjets, model_SR_like_wjets, global_ff_wjets, model_AR_like_qcd, model_SR_like_qcd, global_ff_qcd, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, ff_estimator=args.ff_estimator, prior_ar_over_sr_wjets=prior_ar_over_sr_wjets, prior_ar_over_sr_qcd=prior_ar_over_sr_qcd, classifier_features_wjets=classifier_features_wjets, classifier_features_qcd=classifier_features_qcd, correction_model_wjets_dr=correction_model_wjets_dr, correction_model_wjets_antidr=correction_model_wjets_antidr, correction_prior_ar_over_sr_wjets_dr=correction_prior_ar_over_sr_wjets_dr, correction_prior_ar_over_sr_wjets_antidr=correction_prior_ar_over_sr_wjets_antidr, correction_features_wjets_dr=correction_features_wjets_dr, correction_features_wjets_antidr=correction_features_wjets_antidr, correction_global_ff_wjets_dr=global_ff_wjets_dr_correction, correction_global_ff_wjets_antidr=global_ff_wjets_antidr_correction)
    data_DY_AR_OS_nf, _ = normalizing_flow_ff(data_DY_AR_OS, variables, model_AR_like_wjets, model_SR_like_wjets, global_ff_wjets, model_AR_like_qcd, model_SR_like_qcd, global_ff_qcd, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, ff_estimator=args.ff_estimator, prior_ar_over_sr_wjets=prior_ar_over_sr_wjets, prior_ar_over_sr_qcd=prior_ar_over_sr_qcd, classifier_features_wjets=classifier_features_wjets, classifier_features_qcd=classifier_features_qcd, correction_model_wjets_dr=correction_model_wjets_dr, correction_model_wjets_antidr=correction_model_wjets_antidr, correction_prior_ar_over_sr_wjets_dr=correction_prior_ar_over_sr_wjets_dr, correction_prior_ar_over_sr_wjets_antidr=correction_prior_ar_over_sr_wjets_antidr, correction_features_wjets_dr=correction_features_wjets_dr, correction_features_wjets_antidr=correction_features_wjets_antidr, correction_global_ff_wjets_dr=global_ff_wjets_dr_correction, correction_global_ff_wjets_antidr=global_ff_wjets_antidr_correction)
    data_ST_AR_OS_nf, _ = normalizing_flow_ff(data_ST_AR_OS, variables, model_AR_like_wjets, model_SR_like_wjets, global_ff_wjets, model_AR_like_qcd, model_SR_like_qcd, global_ff_qcd, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, ff_estimator=args.ff_estimator, prior_ar_over_sr_wjets=prior_ar_over_sr_wjets, prior_ar_over_sr_qcd=prior_ar_over_sr_qcd, classifier_features_wjets=classifier_features_wjets, classifier_features_qcd=classifier_features_qcd, correction_model_wjets_dr=correction_model_wjets_dr, correction_model_wjets_antidr=correction_model_wjets_antidr, correction_prior_ar_over_sr_wjets_dr=correction_prior_ar_over_sr_wjets_dr, correction_prior_ar_over_sr_wjets_antidr=correction_prior_ar_over_sr_wjets_antidr, correction_features_wjets_dr=correction_features_wjets_dr, correction_features_wjets_antidr=correction_features_wjets_antidr, correction_global_ff_wjets_dr=global_ff_wjets_dr_correction, correction_global_ff_wjets_antidr=global_ff_wjets_antidr_correction)
    data_ttbar_L_AR_OS_nf, _ = normalizing_flow_ff(data_ttbar_L_AR_OS, variables, model_AR_like_wjets, model_SR_like_wjets, global_ff_wjets, model_AR_like_qcd, model_SR_like_qcd, global_ff_qcd, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, ff_estimator=args.ff_estimator, prior_ar_over_sr_wjets=prior_ar_over_sr_wjets, prior_ar_over_sr_qcd=prior_ar_over_sr_qcd, classifier_features_wjets=classifier_features_wjets, classifier_features_qcd=classifier_features_qcd, correction_model_wjets_dr=correction_model_wjets_dr, correction_model_wjets_antidr=correction_model_wjets_antidr, correction_prior_ar_over_sr_wjets_dr=correction_prior_ar_over_sr_wjets_dr, correction_prior_ar_over_sr_wjets_antidr=correction_prior_ar_over_sr_wjets_antidr, correction_features_wjets_dr=correction_features_wjets_dr, correction_features_wjets_antidr=correction_features_wjets_antidr, correction_global_ff_wjets_dr=global_ff_wjets_dr_correction, correction_global_ff_wjets_antidr=global_ff_wjets_antidr_correction)
    data_embedding_AR_OS_nf, _ = normalizing_flow_ff(data_embedding_AR_OS, variables, model_AR_like_wjets, model_SR_like_wjets, global_ff_wjets, model_AR_like_qcd, model_SR_like_qcd, global_ff_qcd, device, plotting=False, plot_dir=category_plot_dir, include_njets=include_njets_feature, ff_estimator=args.ff_estimator, prior_ar_over_sr_wjets=prior_ar_over_sr_wjets, prior_ar_over_sr_qcd=prior_ar_over_sr_qcd, classifier_features_wjets=classifier_features_wjets, classifier_features_qcd=classifier_features_qcd, correction_model_wjets_dr=correction_model_wjets_dr, correction_model_wjets_antidr=correction_model_wjets_antidr, correction_prior_ar_over_sr_wjets_dr=correction_prior_ar_over_sr_wjets_dr, correction_prior_ar_over_sr_wjets_antidr=correction_prior_ar_over_sr_wjets_antidr, correction_features_wjets_dr=correction_features_wjets_dr, correction_features_wjets_antidr=correction_features_wjets_antidr, correction_global_ff_wjets_dr=global_ff_wjets_dr_correction, correction_global_ff_wjets_antidr=global_ff_wjets_antidr_correction)

    data_events = data_SR_OS[(data_SR_OS.process == 0)]
    data_diboson_SR_OS = data_SR_OS[(data_SR_OS.process == 2) | (data_SR_OS.process == 3)]
    data_DY_SR_OS = data_SR_OS[(data_SR_OS.process == 4) | (data_SR_OS.process == 5)]
    data_ST_SR_OS = data_SR_OS[(data_SR_OS.process == 6) | (data_SR_OS.process == 7)]
    data_ttbar_L_SR_OS = data_SR_OS[(data_SR_OS.process == 9)]
    data_embedding_SR_OS = data_SR_OS[(data_SR_OS.process == 10)]

    data_AR_OS_classic = total_ff_corrected(data_AR_OS)
    data_diboson_AR_OS_classic = total_ff_corrected(data_diboson_AR_OS)
    data_DY_AR_OS_classic = total_ff_corrected(data_DY_AR_OS)
    data_ST_AR_OS_classic = total_ff_corrected(data_ST_AR_OS)
    data_ttbar_L_AR_OS_classic = total_ff_corrected(data_ttbar_L_AR_OS)
    data_embedding_AR_OS_classic = total_ff_corrected(data_embedding_AR_OS)

    total_variables = len(list_variables)
    for index, (var, bins, xlabel) in enumerate(zip(list_variables, list_bins, list_xlabels), start=1):
        if should_log_plot_progress(index, total_variables):
            logger.info(
                "Plotting %s: %d/%d variables (%s)",
                category_name,
                index,
                total_variables,
                var,
            )

        counts_ff_data_classic, bin_edges = np.histogram(data_AR_OS_classic[var], weights=data_AR_OS_classic['corrected_ff'], bins=bins)
        counts_ff_data_classic2, _ = np.histogram(data_AR_OS_classic[var], weights=data_AR_OS_classic['corrected_ff']**2, bins=bins)

        counts_ff_diboson_classic, _ = np.histogram(data_diboson_AR_OS_classic[var], weights=data_diboson_AR_OS_classic.weight * data_diboson_AR_OS_classic['corrected_ff'], bins=bins)
        counts_ff_diboson_classic2, _ = np.histogram(data_diboson_AR_OS_classic[var], weights=(data_diboson_AR_OS_classic.weight * data_diboson_AR_OS_classic['corrected_ff'])**2, bins=bins)
        counts_ff_DY_classic, _ = np.histogram(data_DY_AR_OS_classic[var], weights=data_DY_AR_OS_classic.weight * data_DY_AR_OS_classic['corrected_ff'], bins=bins)
        counts_ff_DY_classic2, _ = np.histogram(data_DY_AR_OS_classic[var], weights=(data_DY_AR_OS_classic.weight * data_DY_AR_OS_classic['corrected_ff'])**2, bins=bins)
        counts_ff_ST_classic, _ = np.histogram(data_ST_AR_OS_classic[var], weights=data_ST_AR_OS_classic.weight * data_ST_AR_OS_classic['corrected_ff'], bins=bins)
        counts_ff_ST_classic2, _ = np.histogram(data_ST_AR_OS_classic[var], weights=(data_ST_AR_OS_classic.weight * data_ST_AR_OS_classic['corrected_ff'])**2, bins=bins)
        counts_ff_ttbar_L_classic, _ = np.histogram(data_ttbar_L_AR_OS_classic[var], weights=data_ttbar_L_AR_OS_classic.weight * data_ttbar_L_AR_OS_classic['corrected_ff'], bins=bins)
        counts_ff_ttbar_L_classic2, _ = np.histogram(data_ttbar_L_AR_OS_classic[var], weights=(data_ttbar_L_AR_OS_classic.weight * data_ttbar_L_AR_OS_classic['corrected_ff'])**2, bins=bins)
        counts_ff_embedding_classic, _ = np.histogram(data_embedding_AR_OS_classic[var], weights=data_embedding_AR_OS_classic.weight * data_embedding_AR_OS_classic['corrected_ff'], bins=bins)
        counts_ff_embedding_classic2, _ = np.histogram(data_embedding_AR_OS_classic[var], weights=(data_embedding_AR_OS_classic.weight * data_embedding_AR_OS_classic['corrected_ff'])**2, bins=bins)

        counts_FF_classic = counts_ff_data_classic - counts_ff_diboson_classic - counts_ff_DY_classic - counts_ff_ST_classic - counts_ff_ttbar_L_classic - counts_ff_embedding_classic

        counts_ff_data, bin_edges = np.histogram(data_AR_OS_nf[var], weights=data_AR_OS_nf['ff_nf'], bins=bins)
        counts_ff_data2, _ = np.histogram(data_AR_OS_nf[var], weights=data_AR_OS_nf['ff_nf']**2, bins=bins)

        counts_ff_diboson, _ = np.histogram(data_diboson_AR_OS_nf[var], weights=data_diboson_AR_OS_nf.weight * data_diboson_AR_OS_nf['ff_nf'], bins=bins)
        counts_ff_diboson2, _ = np.histogram(data_diboson_AR_OS_nf[var], weights=(data_diboson_AR_OS_nf.weight * data_diboson_AR_OS_nf['ff_nf'])**2, bins=bins)
        counts_ff_DY, _ = np.histogram(data_DY_AR_OS_nf[var], weights=data_DY_AR_OS_nf.weight * data_DY_AR_OS_nf['ff_nf'], bins=bins)
        counts_ff_DY2, _ = np.histogram(data_DY_AR_OS_nf[var], weights=(data_DY_AR_OS_nf.weight * data_DY_AR_OS_nf['ff_nf'])**2, bins=bins)
        counts_ff_ST, _ = np.histogram(data_ST_AR_OS_nf[var], weights=data_ST_AR_OS_nf.weight * data_ST_AR_OS_nf['ff_nf'], bins=bins)
        counts_ff_ST2, _ = np.histogram(data_ST_AR_OS_nf[var], weights=(data_ST_AR_OS_nf.weight * data_ST_AR_OS_nf['ff_nf'])**2, bins=bins)
        counts_ff_ttbar_L, _ = np.histogram(data_ttbar_L_AR_OS_nf[var], weights=data_ttbar_L_AR_OS_nf.weight * data_ttbar_L_AR_OS_nf['ff_nf'], bins=bins)
        counts_ff_ttbar_L2, _ = np.histogram(data_ttbar_L_AR_OS_nf[var], weights=(data_ttbar_L_AR_OS_nf.weight * data_ttbar_L_AR_OS_nf['ff_nf'])**2, bins=bins)
        counts_ff_embedding, _ = np.histogram(data_embedding_AR_OS_nf[var], weights=data_embedding_AR_OS_nf.weight * data_embedding_AR_OS_nf['ff_nf'], bins=bins)
        counts_ff_embedding2, _ = np.histogram(data_embedding_AR_OS_nf[var], weights=(data_embedding_AR_OS_nf.weight * data_embedding_AR_OS_nf['ff_nf'])**2, bins=bins)
        counts_FF = counts_ff_data - counts_ff_diboson - counts_ff_DY - counts_ff_ST - counts_ff_ttbar_L - counts_ff_embedding

        counts_diboson, _ = np.histogram(data_diboson_SR_OS[var], weights=data_diboson_SR_OS.weight, bins=bins)
        counts_diboson2, _ = np.histogram(data_diboson_SR_OS[var], weights=data_diboson_SR_OS.weight**2, bins=bins)
        counts_DY, _ = np.histogram(data_DY_SR_OS[var], weights=data_DY_SR_OS.weight, bins=bins)
        counts_DY2, _ = np.histogram(data_DY_SR_OS[var], weights=data_DY_SR_OS.weight**2, bins=bins)
        counts_ST, _ = np.histogram(data_ST_SR_OS[var], weights=data_ST_SR_OS.weight, bins=bins)
        counts_ST2, _ = np.histogram(data_ST_SR_OS[var], weights=data_ST_SR_OS.weight**2, bins=bins)
        counts_ttbar_L, _ = np.histogram(data_ttbar_L_SR_OS[var], weights=data_ttbar_L_SR_OS.weight, bins=bins)
        counts_ttbar_L2, _ = np.histogram(data_ttbar_L_SR_OS[var], weights=data_ttbar_L_SR_OS.weight**2, bins=bins)
        counts_embedding, _ = np.histogram(data_embedding_SR_OS[var], weights=data_embedding_SR_OS.weight, bins=bins)
        counts_embedding2, _ = np.histogram(data_embedding_SR_OS[var], weights=data_embedding_SR_OS.weight**2, bins=bins)

        counts_data, _ = np.histogram(data_events[var], bins=bins)

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
        CMS_NJETS_TITLE(ax, title=njets_title)

        y_error = np.sqrt(counts_data)
        x_error = 0.5 * bin_widths
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

        y_error_stat = np.divide(num, den, out=np.zeros_like(num), where=den != 0)

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
            where=den_classic != 0
        )

        stack_components = [
            (counts_diboson, "#94a4a2", 'Diboson'),
            (counts_ttbar_L, '#832db6', r'$t\bar{t} \to \tau$'),
            (counts_ST, "#717581", r"Single t"),
            (counts_DY, '#3f90da', r'$Z \to \ell \ell$'),
            (counts_FF, "#a96b59", r'Jet $\rightarrow \tau_h$'),
            (counts_embedding, '#ffa90e', r'$\tau$ embedded'),
        ]
        counts_stack_total = draw_stacked_stepfill(ax[0], bin_edges, stack_components)
        ax[0].stairs(counts_stack_total, bin_edges, color='black', linewidth=0.7)

        ax[0].errorbar(bin_centers, counts_data, yerr=y_error, xerr=x_error, fmt='o', color='black', label='Data', markersize=6, elinewidth=1.2, capsize=0)
        ax[0].set_ylabel("Events")
        handles, labels = ax[0].get_legend_handles_labels()
        handles = handles[::-1]
        labels = labels[::-1]
        handles, labels = reorder_for_rowwise_legend(handles, labels, ncol=4)
        ax[0].legend(handles, labels, title=' ', title_fontsize=20, loc='upper left', ncol=4, frameon=False)
        adjust_ylim_for_legend(ax[0])
        ax[0].tick_params(direction='in', top=True, right=True)

        ax[1].errorbar(
            bin_centers,
            np.divide(counts_data, den, out=np.zeros_like(counts_data, dtype=float), where=den != 0),
            xerr=x_error,
            yerr=np.divide(y_error, counts_data, out=np.zeros_like(counts_data, dtype=float), where=counts_data != 0),
            fmt='o',
            color='black',
            markersize=6,
            label=(r'NN $F_\text{F}$' if args.ff_estimator == 'binary_classifier' else r'NF $F_\text{F}$')
        )
        ax[1].fill_between(bin_centers, 1 - y_error_stat, 1 + y_error_stat, color="gray", alpha=0.3, step='mid', label="Stat. Unc.")
        ax[1].axhline(1, color='red', linestyle='--', linewidth=1.5)
        ax[1].set_ylabel("Data / Model")
        ax[1].set_ylim([args.ratio_ylim_min, args.ratio_ylim_max])
        ax[1].grid(True, linestyle=':', alpha=0.7)
        ax[1].tick_params(direction='in', top=True, right=True)
        ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=2, frameon=False)

        ax[2].axis('off')

        ax[3].errorbar(
            bin_centers,
            np.divide(counts_data, den_classic, out=np.zeros_like(counts_data, dtype=float), where=den_classic != 0),
            xerr=x_error,
            yerr=np.divide(y_error, counts_data, out=np.zeros_like(counts_data, dtype=float), where=counts_data != 0),
            fmt='o',
            color='black',
            markersize=6,
            label=r'Cor class $F_\text{F}$ '
        )
        ax[3].fill_between(bin_centers, 1 - y_error_stat_classic, 1 + y_error_stat_classic, color="gray", alpha=0.3, step='mid', label="Stat. Unc.")
        ax[3].axhline(1, color='red', linestyle='--', linewidth=1.5)
        ax[3].set_ylabel("Data / Model")
        ax[3].set_ylim([args.ratio_ylim_min, args.ratio_ylim_max])
        ax[3].grid(True, linestyle=':', alpha=0.7)
        ax[3].tick_params(direction='in', top=True, right=True)
        ax[3].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=2, frameon=False)
        ax[3].set_xlabel(xlabel)

        fig.savefig(category_plot_dir / f'{var}.png')
        fig.savefig(category_plot_dir / f'{var}.pdf')
        plt.close(fig)

        # Plot AR data with clipping information if requested
        if args.plot_ar_data_with_clipping:
            plot_ar_data_with_clipping_info(
                var=var,
                bins=bins,
                xlabel=xlabel,
                data_ar_os_full=data_AR_OS,
                clipping_mask=ar_os_clipping_mask,
                njets_title=njets_title,
                output_dir=category_plot_dir,
            )

    logger.info("Finished %s: saved plots to %s", category_name, category_plot_dir)

def plot_AR_SR_like_alone(model_SR_like, model_AR_like, X, variable, df):
    if args.ff_estimator != 'binary_classifier':
        logger.warning('Skipping AR/SR-like PDF plot: only supported for nf FF estimator.')
        return
    pdf_SR_like = evaluate_pdf(model_SR_like, X)
    pdf_AR_like = evaluate_pdf(model_AR_like, X)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pdf_SR_like, df[variable], label = 'SR-like PDF', color='blue')
    ax.plot(pdf_AR_like, df[variable], label = 'AR-like PDF', color='red')
    ax.set_xlabel(variable)
    ax.set_ylabel('PDF')
    ax.legend()
    plt.show()


if args.plot_taylor_coefficients:
    if args.ff_estimator == 'binary_classifier':
        logger.warning('Skipping Taylor plots: only supported for NF models.')
    else:
        plot_nf_taylor_analysis(plot_root_dir / 'taylor_analysis')
        plot_nf_second_order_covariance(plot_root_dir / 'taylor_analysis')

njets_categories = [
    ('njets_0', r'$N_{jets} = 0$'),
    ('njets_1', r'$N_{jets} = 1$'),
    ('njets_geq_2', r'$N_{jets} \geq 2$'),
    ('njets_inclusive', r'$N_{jets} \geq 0$'),
]

for category_name, njets_title in njets_categories:
    logger.info("Queueing plot production for %s", category_name)
    if args.plot_ff_results or (args.plot_nf_sampling and args.ff_estimator == 'nf'):
        run_plots_for_njets_category(category_name, njets_title)

logger.info("Completed all njets plot categories")
