import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import logging
import yaml
from pathlib import Path
import seaborn as sns
import sys
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / 'configs'
DATA_DIR = WORKSPACE_ROOT / 'data'
MASKS_CONFIG_PATH = CONFIG_DIR / 'masks_MC.yaml'



if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from classes.NeuralNetworks import ConditionalRealNVP, ConditionalFlow1D
from classes.Dataclasses import _component_collection
from classes.Collection import load_model_config, ModelConfig
from classes.Collection import compute_eventwise_fake_factors, load_conditional_flow
from CustomLogging import setup_logging, LogContext


def resolve_training_name(variables: list[str]) -> str:
    tail = variables[4:]
    tag = '_'.join(tail) if tail else 'none'
    return f"training_vars{len(variables)}_{tag}"

logger = setup_logging(logger=logging.getLogger(__name__))

mode_dir = 'conditional_njets_input'
OUTPUT_ROOT = SCRIPT_DIR / 'Training_results_MC'

config_path = CONFIG_DIR / 'config_NF.yaml'



with open(CONFIG_DIR / 'training_variables.yaml', 'r') as f:
    training_variables_cfg = yaml.safe_load(f)
    variables = training_variables_cfg['variables_MC']
    variables_correction = training_variables_cfg['variables_correction']
    variables_correlation = training_variables_cfg['variables_correlation']


dim = len(variables)

resolved_tag = resolve_training_name(variables)
correction_tag = resolve_training_name(variables_correction)
# Training models are in the original Training_results_MC directory
TRAINING_ROOT = SCRIPT_DIR / 'Training_results_MC' / mode_dir / resolved_tag
# Output results folder depends on trained variables
OUTPUT_ROOT = SCRIPT_DIR / f'Training_results_MC_{resolved_tag}'
PLOTS_DIR = OUTPUT_ROOT / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CORR_PLOTS_DIR = PLOTS_DIR / 'correlation'
CORR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
FF_FACTORS_DIR = SCRIPT_DIR / 'Fake_Factors'
FF_FACTORS_DIR.mkdir(parents=True, exist_ok=True)



def get_my_data(df, training_var):
    _df = df
    return _component_collection(
        X=_df[training_var].to_numpy(dtype=np.float32),
        weights=_df["weight"].to_numpy(dtype=np.float32),
    )
def get_my_data_events(df, training_var):
    _df = df
    return _component_collection(
        X=_df[training_var].to_numpy(dtype=np.float32),
        weights = _df["weight_wjets"].to_numpy(dtype=np.float32),
    )


def load_saved_model_config(checkpoint_dir: str | Path, fallback_path: str | Path) -> ModelConfig:
    saved_config_path = Path(checkpoint_dir).parent / 'config.yaml'
    if saved_config_path.exists():
        with open(saved_config_path, 'r') as handle:
            raw = yaml.unsafe_load(handle)

        if isinstance(raw, ModelConfig):
            return raw

        values = vars(raw) if hasattr(raw, '__dict__') else raw
        # Handle nested format written by yaml.safe_dump: {'model': {...}, 'training': {...}, ...}
        if 'model' in values and isinstance(values['model'], dict):
            values = values['model']
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


def load_masks_config(path: str | Path = MASKS_CONFIG_PATH) -> dict[str, list[str]]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f'Mask configuration not found: {config_path}')

    with open(config_path, 'r') as handle:
        raw = yaml.safe_load(handle) or {}

    masks = raw.get('masks', raw)
    if not isinstance(masks, dict):
        raise ValueError(f'Invalid masks configuration in {config_path}: expected a mapping at top level.')

    normalized: dict[str, list[str]] = {}
    for name, expressions in masks.items():
        if isinstance(expressions, str):
            normalized[name] = [expressions]
        elif isinstance(expressions, list) and all(isinstance(expr, str) for expr in expressions):
            normalized[name] = expressions
        else:
            raise TypeError(
                f"Mask '{name}' in {config_path} must be a string or list[str], got {type(expressions).__name__}."
            )

    logger.info('Loaded %d masks from %s', len(normalized), config_path)
    return normalized

def _build_mask_from_config(
    df: pd.DataFrame,
    mask_name: str,
    masks_config: dict[str, list[str]],
) -> pd.Series:
    expressions = masks_config.get(mask_name)
    if not expressions:
        raise KeyError(f"Mask '{mask_name}' is not defined in masks config.")

    combined_expression = ' & '.join(f'({expr})' for expr in expressions)
    mask = df.eval(combined_expression, engine='python')
    return mask.fillna(False).astype(bool)

def _apply_config_mask(
    df: pd.DataFrame,
    mask_name: str,
    masks_config: dict[str, list[str]],
) -> pd.DataFrame:
    return df[_build_mask_from_config(df, mask_name, masks_config)].copy()

def load_flow_model(
    checkpoint_dir: str | Path,
    device: t.device,
) -> tuple['ConditionalFlow1D', dict]:
    """Load a trained ConditionalFlow1D from a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    scaler_meta_path = checkpoint_dir.parent / 'scaler_meta.yaml'
    if not scaler_meta_path.exists():
        raise FileNotFoundError(f'scaler_meta.yaml not found at {scaler_meta_path}')

    with open(scaler_meta_path, 'r') as f:
        meta = yaml.safe_load(f)

    cond_dim = meta['cond_dim']
    model = ConditionalFlow1D(cond_dim).to(device)
    model.initialize_scaler(
        t.tensor(meta['shift_training'], dtype=t.float32),
        t.tensor(meta['scale_training'], dtype=t.float32),
    )
    model.initialize_cond_scaler(
        t.tensor(meta['shift_cond'], dtype=t.float32),
        t.tensor(meta['scale_cond'], dtype=t.float32),
    )

    checkpoint = t.load(checkpoint_dir / 'model_checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info('Loaded ConditionalFlow1D from %s (epoch %d, val NLL=%.4f)',
                checkpoint_dir, checkpoint.get('epoch', -1), checkpoint.get('val_loss', float('nan')))
    return model, meta


@t.no_grad()
def compute_ff_sr_from_flow(
    model: 'ConditionalFlow1D',
    df: pd.DataFrame,
    ff_dr_col: str,
    flow_variables: list[str],
    device: t.device,
    use_log_transform: bool = False,
    ff_clip_max: float = None,
) -> np.ndarray:
    """Sample FF_SR for each event in df using the trained ConditionalFlow1D.

    Events with FF_DR <= 0 or non-finite receive NaN. Events above ff_clip_max are
    included but their FF_DR is clamped to ff_clip_max before being passed to the flow.
    """
    ff_dr = df[ff_dr_col].to_numpy(dtype='float32')
    valid = np.isfinite(ff_dr) & (ff_dr > 0)

    result = np.full(len(df), np.nan, dtype='float32')
    if not valid.any():
        return result

    df_valid = df[valid]
    ff_dr_valid = ff_dr[valid]
    if ff_clip_max is not None:
        ff_dr_valid = np.clip(ff_dr_valid, None, ff_clip_max)
    cond_ff_dr = np.log(ff_dr_valid) if use_log_transform else ff_dr_valid

    cond_np = np.column_stack([
        cond_ff_dr,
        df_valid[flow_variables].to_numpy(dtype='float32'),
    ])
    cond = t.tensor(cond_np).to(device)
    samples = model.sample(cond, n_samples=1).squeeze(0).cpu().numpy()

    if use_log_transform:
        samples = np.exp(samples)

    result[valid] = samples
    return result


def plot_ff_sr_comparison(
    ff_sr_real: np.ndarray,
    ff_sr_flow: np.ndarray,
    plot_dir: Path = None,
    bins: int = 50,
    range: tuple[float, float] = (0.001, 10),
    title: str = 'FF_SR: real vs flow prediction',
    xlim: float = None,
):
    """Compare real FF_SR (NF density ratio) with flow-predicted FF_SR (from conditionals only)."""
    if plot_dir is None:
        plot_dir = SCRIPT_DIR
    bin_edges = np.logspace(np.log10(range[0]), np.log10(range[1]), bins + 1)

    counts_real, _ = np.histogram(ff_sr_real, bins=bin_edges)
    counts_flow, _ = np.histogram(ff_sr_flow, bins=bin_edges)

    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric bin centres

    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(8, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True
    )

    ax_main.step(bin_edges[:-1], counts_real, where='post', color='blue', label='FF_SR real (NF density ratio)')
    ax_main.step(bin_edges[:-1], counts_flow, where='post', color='orange', label='FF_SR flow (from conditionals)')
    ax_main.set_ylabel('Events')
    ax_main.set_xscale('log')
    ax_main.set_title(title)
    ax_main.legend()
    ax_main.grid(True, linestyle='--', alpha=0.5)
    if xlim is not None:
        ax_main.set_xlim(right=xlim)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(counts_real > 0, counts_flow / counts_real, np.nan)

    ax_ratio.axhline(1.0, color='black', linewidth=0.8, linestyle='--')
    ax_ratio.step(bin_edges[:-1], ratio, where='post', color='orange')
    ax_ratio.set_xlabel('Fake Factor')
    ax_ratio.set_ylabel('Flow / Real')
    ax_ratio.set_ylim(0.0, 2.0)
    ax_ratio.set_xscale('log')
    ax_ratio.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_dir / 'ff_sr_real_vs_flow.png')
    plt.close()


def plot_ff_comparison_with_flow(
    ff_sr_nf: np.ndarray,
    ff_dr: np.ndarray,
    ff_sr_flow: np.ndarray,
    plot_dir: Path = None,
    bins: int = 50,
    range: tuple[float, float] = (0.001, 10),
    title: str = 'Fake Factor Comparison (NF vs Flow)',
    xlim: float = None,
):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR
    bins = np.logspace(np.log10(range[0]), np.log10(range[1]), bins + 1)
    plt.figure(figsize=(8, 6))
    plt.hist(ff_sr_nf,   bins=bins, histtype='step', color='blue',  edgecolor='blue',  label='FF_SR (NF density ratio)')
    plt.hist(ff_dr,      bins=bins, histtype='step', color='red',   edgecolor='red',   label='FF_DR')
    plt.hist(ff_sr_flow, bins=bins, histtype='step', color='green', edgecolor='green', label='FF_SR (flow sample)')
    plt.title(title)
    plt.xlabel('Fake Factor')
    plt.ylabel('Frequency')
    plt.xscale('log')
    if xlim is not None:
        plt.xlim(right=xlim)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / 'fake_factor_comparison_flow.png')
    plt.close()


def compute_fake_factors(
    log_pdf_ar: np.ndarray,
    log_pdf_sr: np.ndarray,
    global_ff: float,
    clip_range: tuple[float, float] | None = (0, 10),
) -> np.ndarray:
    """Event-wise FF from log-density ratio: FF = global_ff * exp(log p_SR - log p_AR)."""
    log_ratio = np.clip(log_pdf_sr - log_pdf_ar, -700.0, 700.0)
    ff = global_ff * np.exp(log_ratio)
    if clip_range is None:
        return ff
    return np.clip(ff, clip_range[0], clip_range[1])


def add_fake_factors_to_feather(
    full_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    ff_dr: np.ndarray,
    ff_sr: np.ndarray,
    feather_path: str | Path,
    ff_dr_column: str = 'FF_DR',
    ff_sr_column: str = 'FF_SR',
) -> pd.DataFrame:
    """Attach FF columns to full_df at selected_df indices and persist to feather."""
    if len(selected_df) != len(ff_dr) or len(selected_df) != len(ff_sr):
        raise ValueError(
            'Length mismatch while writing fake factors: '
            f'selected_df={len(selected_df)}, ff_dr={len(ff_dr)}, ff_sr={len(ff_sr)}'
        )

    out_df = full_df.copy()
    if ff_dr_column not in out_df.columns:
        out_df[ff_dr_column] = np.nan
    if ff_sr_column not in out_df.columns:
        out_df[ff_sr_column] = np.nan

    out_df.loc[selected_df.index, ff_dr_column] = ff_dr.astype(np.float32)
    out_df.loc[selected_df.index, ff_sr_column] = ff_sr.astype(np.float32)

    feather_path = Path(feather_path)
    out_df.to_feather(feather_path)
    logger.info(
        'Wrote %s and %s for %d rows to %s',
        ff_dr_column,
        ff_sr_column,
        len(selected_df),
        feather_path,
    )
    return out_df

@t.no_grad()
def evaluate_log_pdf(model: ConditionalRealNVP, X: t.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return log-pdf and preprocessing-valid mask for each event."""
    cond_dim = int(getattr(model, 'cond_dim', 0))
    x_features = X[:, cond_dim:]
    x_preprocessed, log_det_preprocess, valid_mask = model.apply_preprocessing(x_features)
    Xs = model.apply_scaler(x_preprocessed)

    n_invalid_preprocess = (~valid_mask).sum().item()
    n_nan_after_scale    = t.isnan(Xs[valid_mask]).any(dim=-1).sum().item() if valid_mask.any() else 0
    n_inf_after_scale    = t.isinf(Xs[valid_mask]).any(dim=-1).sum().item() if valid_mask.any() else 0
    logger.debug(
        'evaluate_log_pdf: n=%d  invalid_preprocess=%d  nan_after_scale=%d  inf_after_scale=%d',
        X.shape[0], n_invalid_preprocess, n_nan_after_scale, n_inf_after_scale,
    )

    # model(X) returns log-pdf; invalid events are -inf by construction.
    log_pdf = model(X)

    n_nan_logpdf = t.isnan(log_pdf).sum().item()
    n_inf_logpdf = t.isinf(log_pdf).sum().item()
    if n_nan_logpdf > 0:
        logger.warning('evaluate_log_pdf: %d NaN values in log_pdf (valid events with numerical issues)', n_nan_logpdf)
    if n_inf_logpdf > 0:
        logger.debug('evaluate_log_pdf: %d -inf values in log_pdf (invalid preprocess events)', n_inf_logpdf)

    log_pdf = t.nan_to_num(log_pdf, nan=-700.0, neginf=-700.0, posinf=700.0)
    return log_pdf.detach().cpu().numpy(), valid_mask.detach().cpu().numpy()



def plot_ff_distributions(
    ff_SR: np.ndarray,
    ff_DR: np.ndarray,
    plot_dir: Path = None,
    bins: int = 50,
    range: tuple[float, float] = (0.001, 10),
    title: str = "Fake Factor Distribution",
    xlabel: str = "Fake Factor",
    ylabel: str = "Frequency",
):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR
    bins = np.logspace(np.log10(range[0]), np.log10(range[1]), bins + 1)
    plt.figure(figsize=(8, 6))
    plt.hist(ff_SR, bins=bins, histtype='step', alpha=0.5, color='blue', edgecolor='blue', label='SR')
    plt.hist(ff_DR, bins=bins, histtype='step', alpha=0.5, color='red', edgecolor='red', label='DR')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "fake_factor_distribution.png")

def plot_pdf_distribution(
        log_pdf_SR: np.ndarray,
        log_pdf_AR: np.ndarray,
        plot_dir: Path = None,
        bins: int = 50,
        range: tuple[float, float] = (-30, 1),
        title: str = "PDF Distribution",

):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR
    if title == 'DR':
        label_SR = 'SR-like'
        label_AR = 'AR-like'
    else:
        label_SR = 'SR'
        label_AR = 'AR'
    bins = np.linspace(range[0], range[1], bins + 1)
    plt.figure(figsize=(8, 6))
    plt.hist(log_pdf_SR, bins=bins, histtype='step', alpha=0.5, color='blue', edgecolor='blue', label=label_SR)
    plt.hist(log_pdf_AR, bins=bins, histtype='step', alpha=0.5, color='red', ls='--', edgecolor='red', label=label_AR)
    plt.title(title)
    plt.xlabel("log PDF Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if title == 'DR':
        plt.savefig(plot_dir / "pdf_distribution_DR.png")
    elif title == 'SR':
        plt.savefig(plot_dir / "pdf_distribution_SR.png")
    else:
        plt.savefig(plot_dir / "pdf_distribution.png")

def plt_control_plots(
        SR = pd.DataFrame,
        AR = pd.DataFrame,
        FF = np.ndarray,
        var = str,
        range = tuple[float, float],
        region = str,
        plot_dir: Path = None,
):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR

    clipping_mask = FF < 2.0
    correction_factor = len(clipping_mask)/np.sum(np.abs(clipping_mask - 1))

    bins = np.linspace(range[0], range[1], 50)
    plt.figure(figsize=(8, 6))
    plt.hist(SR[var], bins=bins, weights=SR['weight'], histtype='step', color='blue', edgecolor='blue', label='MC')
    plt.hist(AR[var][clipping_mask], bins=bins, weights=correction_factor * FF[clipping_mask] * AR['weight'][clipping_mask], histtype='step', color='red', edgecolor='red', label='FF')
    plt.title(f"Distribution of {var} in {region}")
    plt.xlabel(var)
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{var}_distribution_{region}.png")


def plot_SR_DR_correlation(
        FF_SR,
        FF_DR,
        plot_dir: Path = None,
):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR
    FF_DR = np.clip(FF_DR, 1e-4, None)
    FF_SR = np.clip(FF_SR, 1e-4, None)
    plt.figure(figsize=(5, 5))
    plt.scatter(FF_DR, FF_SR, s=5, alpha=0.3)
    plt.xlabel("FF_DR")
    plt.ylabel("FF_SR")
    plt.title("Baseline FF_SR vs FF_DR (MC)")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / 'ff_sr_vs_ff_dr_scatter.png')
    plt.close()


def colored_scatter(x, y, color, label, plot_dir: Path = None, vmin: float = None, vmax: float = None):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR
    safe_label = str(label).replace(' ', '_').replace('/', '_')
    x = np.clip(x, 1e-4, None)
    y = np.clip(y, 1e-4, None)
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(x, y, c=color, cmap="viridis", s=6, alpha=0.6, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, label=label)
    plt.xlabel("FF_DR")
    plt.ylabel("FF_SR")
    plt.title(f"FF_SR vs FF_DR colored by {label}")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f'ff_sr_vs_ff_dr_colored_{safe_label}.png')
    plt.close()

def slice_plot(
    X,
    X_name,
    n_bins,
    ff_SR_events,
    ffdr_bin_index,
    ffdr_bins,
    x_range: tuple[float, float] | None = None,
    plot_dir: Path = None,
    n_x_bins: int = 25,
):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR

    safe_name = str(X_name).replace(' ', '_').replace('/', '_')

    plt.figure(figsize=(6, 4))

    has_lines = False

    # Define X bins globally (important for comparability). Use configured bounds when provided.
    if x_range is not None and x_range[0] is not None and x_range[1] is not None:
        x_min, x_max = x_range
    else:
        x_min, x_max = np.min(X), np.max(X)
    x_bins = np.linspace(x_min, x_max, n_x_bins + 1)
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])

    for b in range(n_bins):
        mask_dr = ffdr_bin_index == b
        if np.sum(mask_dr) < 200:
            continue

        lo = ffdr_bins[b]
        hi = ffdr_bins[b + 1]

        means = []
        errors = []
        centers = []

        for xb_lo, xb_hi, xc in zip(x_bins[:-1], x_bins[1:], x_centers):
            mask_x = (X >= xb_lo) & (X < xb_hi)
            mask = mask_dr & mask_x

            if np.sum(mask) < 50:
                continue

            vals = ff_SR_events[mask]
            means.append(np.mean(vals))
            errors.append(np.std(vals) / np.sqrt(len(vals)))
            centers.append(xc)

        if len(centers) == 0:
            continue

        plt.errorbar(
            centers,
            means,
            yerr=errors,
            marker='o',
            linestyle='',
            capsize=3,
            label=f"FF_DR [{lo:.3f}, {hi:.3f})"
        )

        has_lines = True

    plt.xlabel(X_name)
    plt.ylabel("FF_SR")
    plt.title(f"FF_SR vs {X_name} in FF_DR slices")
    plt.xlim(x_min, x_max)

    if has_lines:
        plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f'ff_sr_slices_fixed_ff_dr_{safe_name}.png')
    plt.close()

def slice_plot_reg(
    X,
    X_name,
    n_bins,
    ff_SR_events,
    ffdr_bin_index,
    ffdr_bins,
    x_range: tuple[float, float] | None = None,
    plot_dir: Path = None,
    n_x_bins: int = 25,
):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR
    safe_name = str(X_name).replace(' ', '_').replace('/', '_')

    plt.figure(figsize=(6, 4))

    # Use same x-binning strategy as slice_plot for comparability
    if x_range is not None and x_range[0] is not None and x_range[1] is not None:
        x_min, x_max = x_range
    else:
        x_min, x_max = np.min(X), np.max(X)
    x_bins = np.linspace(x_min, x_max, n_x_bins + 1)
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])

    has_lines = False
    for b in range(n_bins):
        mask_dr = ffdr_bin_index == b
        if np.sum(mask_dr) < 200:
            continue

        lo = ffdr_bins[b]
        hi = ffdr_bins[b + 1]

        means = []
        centers = []

        for xb_lo, xb_hi, xc in zip(x_bins[:-1], x_bins[1:], x_centers):
            mask_x = (X >= xb_lo) & (X < xb_hi)
            mask = mask_dr & mask_x

            if np.sum(mask) < 50:
                continue

            means.append(np.mean(ff_SR_events[mask]))
            centers.append(xc)

        if len(centers) < 2:
            continue

        sns.regplot(
            x=np.asarray(centers),
            y=np.asarray(means),
            scatter=False,
            ci=None,
            label=f"FF_DR [{lo:.3f}, {hi:.3f})"
        )
        has_lines = True

    plt.xlabel(X_name)
    plt.ylabel("FF_SR")
    plt.title(f"FF_SR vs {X_name} in FF_DR slices")
    plt.xlim(x_min, x_max)
    if has_lines:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f'ff_sr_slices_fixed_ff_dr_{safe_name}_reg.png')
    plt.close()


def plot_2d_hist_mean_ff_dr(
    X,
    X_name,
    ff_sr,
    ff_dr,
    x_range: tuple[float, float] | None = None,
    plot_dir: Path = None,
    n_x_bins: int = 10,
    n_y_bins: int = 10,
):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR

    safe_name = str(X_name).replace(' ', '_').replace('/', '_')

    if x_range is not None and x_range[0] is not None and x_range[1] is not None:
        x_min, x_max = x_range
    else:
        x_min, x_max = np.min(X), np.max(X)

    y_log_min, y_log_max = 1e-3, 1.0

    x = np.asarray(X)
    y = np.asarray(ff_sr)
    z = np.asarray(ff_dr)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (x >= x_min)
        & (x <= x_max)
        & (y >= y_log_min)
        & (y <= y_log_max)
    )

    x = x[valid]
    y = y[valid]
    z = z[valid]

    if len(x) == 0:
        logger.warning('No valid entries for 2D FF map of %s; skipping.', X_name)
        return

    x_edges = np.linspace(x_min, x_max, n_x_bins + 1)
    y_edges = np.logspace(np.log10(y_log_min), np.log10(y_log_max), n_y_bins + 1)

    sum_ff_dr, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=z)
    count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_ff_dr = np.where(count > 0, sum_ff_dr / count, np.nan)

    import matplotlib.colors as mcolors
    valid_vals = mean_ff_dr[np.isfinite(mean_ff_dr) & (mean_ff_dr > 0)]
    norm = mcolors.LogNorm(
        vmin=valid_vals.min() if len(valid_vals) > 0 else 1e-3,
        vmax=valid_vals.max() if len(valid_vals) > 0 else 1.0,
    )
    plt.figure(figsize=(7, 5))
    mesh = plt.pcolormesh(x_edges, y_edges, mean_ff_dr.T, shading='auto', cmap='viridis', norm=norm)
    cbar = plt.colorbar(mesh)
    cbar.set_label('Mean FF_DR')
    plt.xlabel(X_name)
    plt.ylabel('FF_SR')
    plt.yscale('log')
    plt.title(f'Mean FF_DR in 2D bins: {X_name} vs FF_SR')
    plt.xlim(x_min, x_max)
    plt.ylim(y_log_min, y_log_max)
    plt.tight_layout()
    plt.savefig(plot_dir / f'ffdr_mean_2d_{safe_name}_vs_ffsr.png')
    plt.close()


def plot_2d_hist_event_count_ff_sr_dr(
    ff_sr,
    ff_dr,
    ff_range: tuple[float, float] = (1e-3, 1.0),
    plot_dir: Path = None,
    n_x_bins: int = 30,
    n_y_bins: int = 30,
):
    """Plot FF_SR vs FF_DR as a 2D histogram colored by event counts per bin."""
    if plot_dir is None:
        plot_dir = SCRIPT_DIR

    ff_min, ff_max = ff_range

    x = np.asarray(ff_dr)
    y = np.asarray(ff_sr)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= ff_min)
        & (x <= ff_max)
        & (y >= ff_min)
        & (y <= ff_max)
    )

    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        logger.warning('No valid entries for FF_SR vs FF_DR 2D event-count map; skipping.')
        return

    x_edges = np.logspace(np.log10(ff_min), np.log10(ff_max), n_x_bins + 1)
    y_edges = np.logspace(np.log10(ff_min), np.log10(ff_max), n_y_bins + 1)

    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    import matplotlib.colors as mcolors
    positive_counts = counts[counts > 0]
    norm = mcolors.LogNorm(
        vmin=positive_counts.min() if len(positive_counts) > 0 else 1.0,
        vmax=positive_counts.max() if len(positive_counts) > 0 else 1.0,
    )

    plt.figure(figsize=(7, 5))
    mesh = plt.pcolormesh(x_edges, y_edges, counts.T, shading='auto', cmap='viridis', norm=norm)
    cbar = plt.colorbar(mesh)
    cbar.set_label('Event count')
    plt.xlabel('FF_DR')
    plt.ylabel('FF_SR')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Event count in 2D bins: FF_DR vs FF_SR')
    plt.xlim(ff_min, ff_max)
    plt.ylim(ff_min, ff_max)
    plt.tight_layout()
    plt.savefig(plot_dir / 'ffsr_ffdr_event_count_2d.png')
    plt.close()


def main():

    masks_config = load_masks_config(MASKS_CONFIG_PATH)

    training_root = TRAINING_ROOT

    device = t.device("cuda:1" if t.cuda.is_available() else "cpu")

    chk_pth_model_AR_like = training_root / 'Wjets' / 'DR' / 'AR-like' / 'latest'
    chk_pth_model_SR_like = training_root / 'Wjets' / 'DR' / 'SR-like' / 'latest'
    chk_pth_model_AR = training_root / 'Wjets' / 'antiDR' / 'AR' / 'latest'
    chk_pth_model_SR = training_root / 'Wjets' / 'antiDR' / 'SR' / 'latest'
        
    config_AR_like = load_saved_model_config(chk_pth_model_AR_like, config_path)
    config_SR_like = load_saved_model_config(chk_pth_model_SR_like, config_path)
    config_AR = load_saved_model_config(chk_pth_model_AR, config_path)
    config_SR = load_saved_model_config(chk_pth_model_SR, config_path)

    model_AR_like = load_conditional_flow(dim=dim, cfg=config_AR_like, checkpoint_path=chk_pth_model_AR_like / 'model_checkpoint.pth', device=device)
    model_SR_like = load_conditional_flow(dim=dim, cfg=config_SR_like, checkpoint_path=chk_pth_model_SR_like / 'model_checkpoint.pth', device=device)
    model_AR = load_conditional_flow(dim=dim, cfg=config_AR, checkpoint_path=chk_pth_model_AR / 'model_checkpoint.pth', device=device)
    model_SR = load_conditional_flow(dim=dim, cfg=config_SR, checkpoint_path=chk_pth_model_SR / 'model_checkpoint.pth', device=device)


    data_complete = pd.read_feather(DATA_DIR / 'data_complete.feather')
    data_MC = data_complete[data_complete['process'] == 1]
    data_events = data_complete[data_complete['process'] == 0]

    data_presel_MC = _apply_config_mask(data_MC, 'mask_preselection_loose', masks_config)
    data_presel_events = _apply_config_mask(data_events, 'mask_preselection_loose', masks_config)

    data_SR_MC = _apply_config_mask(data_presel_MC, 'SR', masks_config)
    data_AR_MC = _apply_config_mask(data_presel_MC, 'AR', masks_config)

    data_SR_events =_apply_config_mask(data_presel_events, 'SR', masks_config)
    data_AR_events = _apply_config_mask(data_presel_events, 'AR', masks_config) 

    data_SR_like_MC = _apply_config_mask(data_presel_MC, 'SR_like', masks_config)
    data_AR_like_MC = _apply_config_mask(data_presel_MC, 'AR_like', masks_config)


    input_variables = ['njets'] + list(variables)

    X_AR_MC = get_my_data(data_AR_MC, input_variables).to_torch(device=None).X.to(device)
    X_AR = get_my_data_events(data_AR_events, input_variables).to_torch(device=None).X.to(device)

    # DR-trained models applied to antiDR (SR region) events

    log_pdf_AR_like, valid_AR_like = evaluate_log_pdf(model_AR_like, X_AR)
    log_pdf_SR_like, valid_SR_like = evaluate_log_pdf(model_SR_like, X_AR)

    # antiDR-trained models applied to DR region (AR-like) events
    log_pdf_AR, valid_AR = evaluate_log_pdf(model_AR, X_AR)
    log_pdf_SR, valid_SR = evaluate_log_pdf(model_SR, X_AR)

    log_pdf_AR_like_MC, valid_AR_like_MC = evaluate_log_pdf(model_AR_like, X_AR_MC)
    log_pdf_SR_like_MC, valid_SR_like_MC = evaluate_log_pdf(model_SR_like, X_AR_MC)

    # antiDR-trained models applied to DR region (AR-like) events
    log_pdf_AR_MC, valid_AR_MC = evaluate_log_pdf(model_AR, X_AR_MC)
    log_pdf_SR_MC, valid_SR_MC = evaluate_log_pdf(model_SR, X_AR_MC)



    global_ff_DR_MC = np.sum(data_SR_like_MC['weight']) / np.sum(data_AR_like_MC['weight'])
    global_ff_SR_MC = np.sum(data_SR_MC['weight']) / np.sum(data_AR_MC['weight'])

    logger.info('Global FF DR: %f', global_ff_DR_MC)
    logger.info('Global FF SR: %f', global_ff_SR_MC)
    logger.info('Valid preprocessing fraction DR: AR-like=%.4f SR-like=%.4f', valid_AR_like.mean(), valid_SR_like.mean())
    logger.info('Valid preprocessing fraction antiDR: AR=%.4f SR=%.4f', valid_AR.mean(), valid_SR.mean())

    ff_DR_events_raw = compute_fake_factors(log_pdf_AR_like, log_pdf_SR_like, global_ff_DR_MC, clip_range=None)
    ff_SR_events_raw = compute_fake_factors(log_pdf_AR, log_pdf_SR, global_ff_SR_MC, clip_range=None)

    ff_DR_events = np.clip(ff_DR_events_raw, 0, 10)
    ff_SR_events = np.clip(ff_SR_events_raw, 0, 10)

    add_fake_factors_to_feather(
        full_df=data_AR_events,
        selected_df=data_AR_events,
        ff_dr=ff_DR_events_raw,
        ff_sr=ff_SR_events_raw,
        feather_path=FF_FACTORS_DIR / f'fake_factors_{resolved_tag}.feather',
    )

    ff_DR_MC = compute_fake_factors(log_pdf_AR_like_MC, log_pdf_SR_like_MC, global_ff_DR_MC)
    ff_SR_MC = compute_fake_factors(log_pdf_AR_MC, log_pdf_SR_MC, global_ff_SR_MC)

    plot_ff_distributions(ff_SR_events_raw, ff_DR_events_raw, title='Fake Factors data events, MC models', plot_dir=PLOTS_DIR)

    plot_pdf_distribution(log_pdf_SR_like, log_pdf_AR_like, plot_dir=PLOTS_DIR, title='DR data')
    plot_pdf_distribution(log_pdf_SR, log_pdf_AR, plot_dir=PLOTS_DIR, title='SR data')

    for var in variables:
        plt_control_plots(SR = data_SR_MC, AR = data_AR_MC, FF = ff_SR_MC, var = var, range=(0, 150), region='DR MC', plot_dir=PLOTS_DIR)
        plt_control_plots(SR = data_SR_MC, AR = data_AR_MC, FF = ff_DR_MC, var=var, range=(0, 150), region='SR MC', plot_dir=PLOTS_DIR)

    # --- Flow-based FF_SR comparison ---
    flow_checkpoint_dir = SCRIPT_DIR / 'FF_flow_results' / resolved_tag / correction_tag / 'FF_SR' / 'latest'
    if flow_checkpoint_dir.exists():
        flow_model, flow_meta = load_flow_model(flow_checkpoint_dir, device=device)
        # Attach the raw FF_DR computed in this run so the flow sees the same values it was trained on
        df_ar_for_flow = data_AR_events.reset_index(drop=True).copy()
        df_ar_for_flow['FF_DR'] = ff_DR_events_raw
        ff_sr_flow = compute_ff_sr_from_flow(
            model=flow_model,
            df=df_ar_for_flow,
            ff_dr_col='FF_DR',
            flow_variables=flow_meta['variables'],
            device=device,
            use_log_transform=flow_meta.get('use_log_transform', False),
            ff_clip_max=flow_meta.get('ff_clip_max', None),
        )
        # Restrict comparison to events where the flow made a prediction (FF_DR in training domain)
        valid_flow = np.isfinite(ff_sr_flow)
        logger.info('Flow comparison: %d / %d events in training domain', valid_flow.sum(), len(valid_flow))
        _ff_clip_max = flow_meta.get('ff_clip_max', None)
        plot_ff_comparison_with_flow(
            ff_sr_nf=ff_SR_events_raw[valid_flow],
            ff_dr=ff_DR_events_raw[valid_flow],
            ff_sr_flow=ff_sr_flow[valid_flow],
            plot_dir=PLOTS_DIR,
            xlim=_ff_clip_max,
        )
        plot_ff_sr_comparison(
            ff_sr_real=ff_SR_events_raw[valid_flow],
            ff_sr_flow=ff_sr_flow[valid_flow],
            plot_dir=PLOTS_DIR,
            xlim=_ff_clip_max,
        )
        logger.info('Flow FF_SR comparison plot saved to %s', PLOTS_DIR)
    else:
        logger.warning('Flow checkpoint not found at %s; skipping flow comparison plot.', flow_checkpoint_dir)

    # ----- correlation stuff -----

    # Sensible color-axis ranges per variable (vmin, vmax)
    color_ranges: dict[str, tuple[float, float]] = {
        'pt_1':              (30,  100),
        'pt_2':              (30,  100),
        'm_vis':             (0,   250),
        'deltaR_ditaupair':  (0.3, 5.0),
        'pt_vis':            (0,   200),
        'met':               (0,   150),
        'pt_tt':             (0,   200),
        'm_fastmtt':         (0,   250),
        'mt_tot':            (0,   200),
        'mt_1':              (0,   70),
    }

    plot_SR_DR_correlation(ff_SR_events, ff_DR_events, plot_dir=CORR_PLOTS_DIR)
    plot_2d_hist_event_count_ff_sr_dr(
        ff_sr=ff_SR_events,
        ff_dr=ff_DR_events,
        plot_dir=CORR_PLOTS_DIR,
    )

    for var in variables_correlation:
        vmin, vmax = color_ranges.get(var, (None, None))
        colored_scatter(ff_DR_events, ff_SR_events, data_AR_events[var], var,
                        plot_dir=CORR_PLOTS_DIR, vmin=vmin, vmax=vmax)



    n_bins = 5
    ffdr_bins = np.percentile(ff_DR_events, np.linspace(0, 100, n_bins + 1))
    ffdr_bin_index = np.digitize(ff_DR_events, ffdr_bins) - 1


    for var in variables_correlation:
        x_range = color_ranges.get(var, (None, None))
        slice_plot(
            X = data_AR_events[var], 
            X_name = var, 
            n_bins = n_bins, 
            ff_SR_events=ff_SR_events,
            ffdr_bin_index=ffdr_bin_index,
            ffdr_bins=ffdr_bins,
            x_range=x_range,
            plot_dir=CORR_PLOTS_DIR,
            )
        slice_plot_reg(
            X = data_AR_events[var], 
            X_name = var, 
            n_bins = n_bins, 
            ff_SR_events=ff_SR_events,
            ffdr_bin_index=ffdr_bin_index,
            ffdr_bins=ffdr_bins,
            x_range=x_range,
            plot_dir=CORR_PLOTS_DIR,
        )
        plot_2d_hist_mean_ff_dr(
            X=data_AR_events[var],
            X_name=var,
            ff_sr=ff_SR_events,
            ff_dr=ff_DR_events,
            x_range=x_range,
            plot_dir=CORR_PLOTS_DIR,
            n_x_bins=10,
            n_y_bins=10,
        )

    # Quantitative residual test
    # 


    reg = LinearRegression()
    reg.fit(ff_DR_events.reshape(-1, 1), ff_SR_events)
    FF_SR_pred = reg.predict(ff_DR_events.reshape(-1, 1))

    residuals = ff_SR_events - FF_SR_pred


    logger.info('Residual correlations (Spearman):')

    importance = {}

    for var in variables_correlation:
        corr, _ = spearmanr(residuals, data_AR_events[var])
        importance[var] = abs(corr)
        logger.info('  %-10s : %+.3f', var, corr)

    plt.figure(figsize=(5, 3))
    plt.bar(importance.keys(), importance.values())
    plt.ylabel('|Residual correlation|')
    plt.title('Candidate variable importance')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(CORR_PLOTS_DIR / 'residual_variable_importance.png')
    plt.close()

        

# ----------

if __name__ == '__main__':
    main()