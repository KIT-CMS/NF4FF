import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import torch as t
import logging
import yaml
from pathlib import Path
import sys

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
from classes.Plotting import CMS_CHANNEL_TITLE, CMS_LABEL, CMS_LUMI_TITLE, CMS_NJETS_TITLE
from CustomLogging import setup_logging, LogContext


REQUIRE_OUT_OF_FOLD_MODELS = True

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

variable_ranges: dict[str, tuple[float, float]] = {
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
    'mt_2':              (0,   150),
    'njets':             (0,    10),
}


def CMS_CHANNEL_TITLE(ax, *args, **kwargs):
    ax[0].set_title(
        r"$e\tau_h$",
        fontsize=20,
        loc="left",
        fontproperties="Tex Gyre Heros"
    )

def CMS_NJETS_TITLE(ax, title=r"$N_{jets} \geq 0$", *args, **kwargs):
    ax[0].set_title(
        title,
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

def CMS_CHANNEL_TITLE2(ax):
    ax.set_title(
        r"$e\tau_h$",
        fontsize=20,
        loc="left",
        fontproperties="Tex Gyre Heros",
        y=1.08,
    )


def CMS_NJETS_TITLE2(ax, title=r"$N_{jets} \geq 0$"):
    ax.set_title(
        title,
        fontsize=20,
        loc="center",
        fontproperties="Tex Gyre Heros",
        y=1.08,
    )


def CMS_LUMI_TITLE2(ax):
    ax.set_title(
        "59.8 $fb^{-1}$ (2018, 13 TeV)",
        fontsize=20,
        loc="right",
        fontproperties="Tex Gyre Heros",
        y=1.08,
    )


def CMS_LABEL(ax, *args, **kwargs):
    ax[0].text(
        0.025, 0.95,
        "Private work (CMS simulation)",
        fontsize=15,
        verticalalignment='top',
        fontproperties="Tex Gyre Heros:italic",
        bbox=dict(facecolor="white", alpha=0, edgecolor="white", boxstyle="round,pad=0.5"),
        transform=ax[0].transAxes
    )

def CMS_LABEL2(fig, ax):
    bbox = ax.get_position()
    fig.text(
        bbox.x0,
        bbox.y1 + 0.015,
        "Private work (CMS simulation)",
        fontsize=15,
        fontproperties="Tex Gyre Heros:italic",
        ha="left",
        va="top"
    )
def resolve_training_name(variables: list[str]) -> str:
    tail = variables[4:]
    tag = '_'.join(tail) if tail else 'none'
    return f"training_vars{len(variables)}_{tag}"

logger = setup_logging(logger=logging.getLogger(__name__))


def _apply_cms_plot_labels(ax) -> None:
    CMS_CHANNEL_TITLE([ax])
    CMS_LUMI_TITLE([ax])
    CMS_LABEL([ax])
    #CMS_NJETS_TITLE([ax], title=r"$N_{jets} \geq 0$")

def _apply_cms_plot_labels_outside(ax) -> None:
    CMS_CHANNEL_TITLE([ax])
    CMS_LUMI_TITLE([ax])
    #CMS_LABEL([ax])
    #CMS_NJETS_TITLE([ax], title=r"$N_{jets} \geq 0$")

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
TRAINING_ROOT = SCRIPT_DIR / 'Training_results_MC' / resolved_tag
# Output results folder depends on trained variables
OUTPUT_ROOT = SCRIPT_DIR / f'Training_results_MC_{resolved_tag}'
PLOTS_DIR = OUTPUT_ROOT / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CORR_PLOTS_DIR = PLOTS_DIR / 'correlation'
CORR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
FF_FACTORS_DIR = SCRIPT_DIR / 'Fake_Factors'
FF_FACTORS_DIR.mkdir(parents=True, exist_ok=True)



def get_my_data(df: pd.DataFrame, training_var: list[str]) -> _component_collection:
    """Return X with event_var prepended: [event_var, njets, feat1, ...]."""
    return _component_collection(
        X=df[['event_var'] + list(training_var)].to_numpy(dtype=np.float32),
        weights=df['weight'].to_numpy(dtype=np.float32),
    )


def get_my_data_events(df: pd.DataFrame, training_var: list[str]) -> _component_collection:
    """Like get_my_data but uses weight_wjets (for data events)."""
    return _component_collection(
        X=df[['event_var'] + list(training_var)].to_numpy(dtype=np.float32),
        weights=df['weight_wjets'].to_numpy(dtype=np.float32),
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

# ──────────────────────────────────────────────
# Model helpers: FoldCombinedNF + unified loader
# ──────────────────────────────────────────────

class FoldCombinedNF(t.nn.Module):
    """Routes log-probability evaluation by event_var.

    even_model: trained on odd events  (event_var == 1) -> used for even events (event_var == 0)
    odd_model:  trained on even events (event_var == 0) -> used for odd  events (event_var == 1)

    Forward input: x[..., 0] = event_var (0 or 1), x[..., 1:] = NF features.
    """

    def __init__(self, even_model: ConditionalRealNVP, odd_model: ConditionalRealNVP) -> None:
        super().__init__()
        self.even_model = even_model
        self.odd_model  = odd_model

    def forward(self, x: t.Tensor) -> t.Tensor:
        even_mask = (x[..., 0].long() == 0).reshape(-1)
        features  = x[..., 1:]
        even_out  = self.even_model(features).reshape(-1)
        odd_out   = self.odd_model(features).reshape(-1)
        return t.where(even_mask, even_out, odd_out)


def _build_sub_model(
    dim: int,
    cfg: ModelConfig,
    state_dict: dict,
    device: t.device,
) -> ConditionalRealNVP:
    """Instantiate a ConditionalRealNVP sub-model and load weights from state_dict."""
    model = ConditionalRealNVP(
        dim=dim,
        cond_dim=1,
        n_layers=cfg.n_layers,
        hidden_dims=(cfg.hidden_dims,),
        s_scale=cfg.s_scale,
        use_cut_preprocessing=cfg.use_cut_preprocessing,
        cut_preprocessing_index=cfg.cut_preprocessing_index,
        cut_preprocessing_thresholds=cfg.cut_preprocessing_thresholds,
        cut_preprocessing_epsilon=cfg.cut_preprocessing_epsilon,
        use_tail_preprocessing=cfg.use_tail_preprocessing,
        tail_preprocessing_index=cfg.tail_preprocessing_index,
        tail_preprocessing_type=cfg.tail_preprocessing_type,
        tail_preprocessing_center=cfg.tail_preprocessing_center,
        tail_preprocessing_scale=cfg.tail_preprocessing_scale,
        tail_preprocessing_epsilon=cfg.tail_preprocessing_epsilon,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_nf_model(
    dim: int,
    cfg: ModelConfig,
    checkpoint_path: Path,
    device: t.device,
    require_folded: bool = REQUIRE_OUT_OF_FOLD_MODELS,
) -> t.nn.Module:
    """Load a ConditionalRealNVP or FoldCombinedNF depending on checkpoint schema."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = t.load(checkpoint_path, map_location=device, weights_only=False)
    schema = checkpoint.get('schema', 'conditional_nf_v1')

    if schema == 'fold_combined_nf_v1':
        even_model = _build_sub_model(dim, cfg, checkpoint['even_model_state_dict'], device)
        odd_model  = _build_sub_model(dim, cfg, checkpoint['odd_model_state_dict'],  device)
        model = FoldCombinedNF(even_model, odd_model).to(device)
        model.eval()
        logger.info('Loaded FoldCombinedNF from %s', checkpoint_path)
        return model

    if require_folded:
        raise RuntimeError(
            'Out-of-fold fake-factor evaluation requires fold-combined checkpoints '
            f"(schema='fold_combined_nf_v1'). Got schema='{schema}' at {checkpoint_path}."
        )

    model = load_conditional_flow(dim=dim, cfg=cfg, checkpoint_path=checkpoint_path, device=device)
    logger.info('Loaded ConditionalRealNVP from %s', checkpoint_path)
    return model


def _validate_event_var_tensor(X: t.Tensor, tensor_name: str) -> None:
    """Validate that event_var exists and is binary (0/1)."""
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError(
            f'{tensor_name}: expected shape [N, 1+features] with event_var in column 0, got {tuple(X.shape)}.'
        )

    event_var = X[:, 0]
    event_var_long = event_var.long()
    if not t.all((event_var_long == 0) | (event_var_long == 1)):
        uniques = t.unique(event_var).detach().cpu().numpy()
        raise ValueError(f'{tensor_name}: event_var must contain only 0/1, got values {uniques}.')

    n_even = int((event_var_long == 0).sum().item())
    n_odd = int((event_var_long == 1).sum().item())
    if n_even == 0 or n_odd == 0:
        logger.warning(
            '%s contains only one fold (event_var==0: %d, event_var==1: %d). Out-of-fold routing still works, '
            'but one sub-model is unused for this dataset.',
            tensor_name,
            n_even,
            n_odd,
        )


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

# ──────────────────────────────────────────────
# Log-PDF evaluation (fold-aware)
# ──────────────────────────────────────────────

@t.no_grad()
def _evaluate_log_pdf_single(
    model: ConditionalRealNVP,
    X: t.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate log-PDF for a single ConditionalRealNVP.

    X: [N, features] — no event_var column; first column is njets if cond_dim==1.
    Returns (log_pdf, valid_mask) as numpy arrays.
    """
    cond_dim   = int(getattr(model, 'cond_dim', 0))
    x_features = X[:, cond_dim:]
    x_preprocessed, _, valid_mask = model.apply_preprocessing(x_features)
    Xs = model.apply_scaler(x_preprocessed)

    n_invalid = (~valid_mask).sum().item()
    n_nan     = t.isnan(Xs[valid_mask]).any(dim=-1).sum().item() if valid_mask.any() else 0
    logger.debug(
        '_evaluate_log_pdf_single: n=%d  invalid_preprocess=%d  nan_after_scale=%d',
        X.shape[0], n_invalid, n_nan,
    )

    log_pdf = model(X)
    if t.isnan(log_pdf).any():
        logger.warning('_evaluate_log_pdf_single: %d NaN values in log_pdf', t.isnan(log_pdf).sum().item())
    log_pdf = t.nan_to_num(log_pdf, nan=-700.0, neginf=-700.0, posinf=700.0)
    return log_pdf.detach().cpu().numpy(), valid_mask.detach().cpu().numpy()


@t.no_grad()
def _evaluate_log_pdf_folded(
    model: FoldCombinedNF,
    X: t.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate log-PDF for a FoldCombinedNF.

    X: [N, 1+features] where X[:, 0] = event_var (0 or 1).
    Routes each event to even_model (event_var==0) or odd_model (event_var==1).
    """
    features   = X[:, 1:]  # [njets, feat1, ...]
    even_bool  = (X[:, 0].long() == 0).cpu().numpy()

    log_pdf    = np.full(X.shape[0], -700.0, dtype=np.float32)
    valid_mask = np.zeros(X.shape[0], dtype=bool)

    for fold_bool, sub_model in [(even_bool, model.even_model), (~even_bool, model.odd_model)]:
        if fold_bool.sum() == 0:
            continue
        fold_idx = t.from_numpy(fold_bool).to(X.device)
        lp, vm = _evaluate_log_pdf_single(sub_model, features[fold_idx])
        log_pdf[fold_bool]    = lp
        valid_mask[fold_bool] = vm

    return log_pdf, valid_mask


@t.no_grad()
def evaluate_log_pdf(
    model: t.nn.Module,
    X: t.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (log_pdf, valid_mask) for each event.

    X: [N, 1+features] with X[:, 0] = event_var.
    Dispatches to the correct evaluator based on model type.
    """
    if isinstance(model, FoldCombinedNF):
        return _evaluate_log_pdf_folded(model, X)
    # Single model: strip event_var column before forwarding
    return _evaluate_log_pdf_single(model, X[:, 1:])



def plot_ff_distributions(
    ff_SR: np.ndarray,
    ff_DR: np.ndarray,
    plot_dir: Path = None,
    bins: int = 50,
    range: tuple[float, float] = (0.0001, 10000),
    title: str = "Fake Factor Distribution",
    xlabel: str = "Fake Factor",
    ylabel: str = "Frequency",
):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR
    ff_SR = np.asarray(ff_SR, dtype=np.float64)
    ff_DR = np.asarray(ff_DR, dtype=np.float64)

    sr_valid = ff_SR[np.isfinite(ff_SR) & (ff_SR > 0.0)]
    dr_valid = ff_DR[np.isfinite(ff_DR) & (ff_DR > 0.0)]
    sr_max = float(np.max(sr_valid)) if sr_valid.size > 0 else np.nan
    dr_max = float(np.max(dr_valid)) if dr_valid.size > 0 else np.nan

    x_min, x_max = float(range[0]), float(range[1])

    bins = np.logspace(np.log10(range[0]), np.log10(range[1]), bins + 1)
    plt.figure(figsize=(8, 6))
    plt.hist(ff_SR, bins=bins, histtype='step', alpha=0.5, color='blue', edgecolor='blue', label='SR')
    plt.hist(ff_DR, bins=bins, histtype='step', alpha=0.5, color='red', edgecolor='red', label='DR')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)

    ax = plt.gca()
    if np.isfinite(sr_max):
        ax.axvline(np.clip(sr_max, x_min, x_max), color='blue', linestyle='--', alpha=0.7, linewidth=1.2)
    if np.isfinite(dr_max):
        ax.axvline(np.clip(dr_max, x_min, x_max), color='red', linestyle='--', alpha=0.7, linewidth=1.2)

    max_text = (
        f'SR max: {sr_max:.3g}\nDR max: {dr_max:.3g}'
        if np.isfinite(sr_max) and np.isfinite(dr_max)
        else 'SR/DR max: n/a'
    )
    ax.text(
        0.03,
        0.97,
        max_text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'),
    )

    plt.legend()
    _apply_cms_plot_labels(ax)
    plt.tight_layout()
    plt.savefig(plot_dir / "fake_factor_distribution.png")
def plt_control_plots(
        SR = pd.DataFrame,
        AR = pd.DataFrame,
        FF = np.ndarray,
        clip_value = float,
        var = str,
        range = tuple[float, float],
        region = str,
        plot_dir: Path = None,
):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR

    ff = np.asarray(FF, dtype=np.float64)
    ar_weight = AR['weight'].to_numpy(dtype=np.float64)
    sr_weight = SR['weight'].to_numpy(dtype=np.float64)

    if len(ff) != len(AR):
        raise ValueError(
            f"Length mismatch in plt_control_plots: len(FF)={len(ff)} vs len(AR)={len(AR)}"
        )

    finite_mask = np.isfinite(ff) & np.isfinite(ar_weight)
    clipping_mask = finite_mask & (ff < clip_value)

    n_total = len(ff)
    n_kept = int(np.sum(clipping_mask))
    n_clipped = n_total - n_kept
    fraction_clipped = (n_clipped / n_total) if n_total > 0 else 0.0

    weighted_sum_before_clip = float(np.sum(ar_weight[finite_mask]))
    weighted_sum_after_clip = float(np.sum(ar_weight[clipping_mask]))

    if weighted_sum_after_clip > 0.0 and np.isfinite(weighted_sum_before_clip):
        correction_factor = weighted_sum_before_clip / weighted_sum_after_clip
    else:
        correction_factor = 1.0
        logger.warning(
            "Using correction_factor=1.0 because weighted_sum_after_clip<=0 or non-finite baseline: "
            "before=%.6e after=%.6e",
            weighted_sum_before_clip,
            weighted_sum_after_clip,
        )

    bins = np.linspace(range[0], range[1], 50)

    sr_counts, _ = np.histogram(SR[var], bins=bins, weights=sr_weight)
    ff_counts_raw, _ = np.histogram(
        AR[var][clipping_mask],
        bins=bins,
        weights=ff[clipping_mask] * ar_weight[clipping_mask],
    )
    ff_counts_renorm, _ = np.histogram(
        AR[var][clipping_mask],
        bins=bins,
        weights=correction_factor * ff[clipping_mask] * ar_weight[clipping_mask],
    )

    sr_integral = float(np.sum(sr_counts))
    ff_integral_raw = float(np.sum(ff_counts_raw))
    ff_integral_renorm = float(np.sum(ff_counts_renorm))
    ratio_raw_to_sr = (ff_integral_raw / sr_integral) if sr_integral != 0.0 else np.nan
    ratio_renorm_to_sr = (ff_integral_renorm / sr_integral) if sr_integral != 0.0 else np.nan

    finite_ff = ff[np.isfinite(ff)]
    if finite_ff.size > 0:
        ff_p95, ff_p99 = np.quantile(finite_ff, [0.95, 0.99])
        ff_max = float(np.max(finite_ff))
    else:
        ff_p95, ff_p99, ff_max = np.nan, np.nan, np.nan

    clipped_weight_fraction = np.nan
    if weighted_sum_before_clip != 0.0 and np.isfinite(weighted_sum_before_clip):
        clipped_weight_fraction = (weighted_sum_before_clip - weighted_sum_after_clip) / weighted_sum_before_clip

    logger.info(
        "Control plot verification (%s, %s): N_total=%d N_kept=%d N_clipped=%d fraction_clipped=%.4f",
        region,
        var,
        n_total,
        n_kept,
        n_clipped,
        fraction_clipped,
    )
    logger.info(
        "Control plot yields (%s, %s): sum_before_clip=%.6e sum_after_clip=%.6e correction_factor=%.6e",
        region,
        var,
        weighted_sum_before_clip,
        weighted_sum_after_clip,
        correction_factor,
    )
    logger.info(
        "Control plot closure (%s, %s): SR=%.6e FF_raw=%.6e FF_renorm=%.6e raw/SR=%.6e renorm/SR=%.6e",
        region,
        var,
        sr_integral,
        ff_integral_raw,
        ff_integral_renorm,
        ratio_raw_to_sr,
        ratio_renorm_to_sr,
    )
    logger.info(
        "FF tail diagnostics (%s, %s): p95=%.6e p99=%.6e max=%.6e clipped_weight_fraction=%.6e",
        region,
        var,
        ff_p95,
        ff_p99,
        ff_max,
        clipped_weight_fraction,
    )

    plt.figure(figsize=(8, 6))
    plt.hist(SR[var], bins=bins, weights=SR['weight'], histtype='step', color='blue', edgecolor='blue', label='MC')
    plt.hist(
        AR[var][clipping_mask],
        bins=bins,
        weights=correction_factor * ff[clipping_mask] * ar_weight[clipping_mask],
        histtype='step',
        color='red',
        edgecolor='red',
        label='FF',
    )
    #plt.title(f"Distribution of {var} in {region}")
    plt.xlabel(var)
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    ax = plt.gca()
    _apply_cms_plot_labels(ax)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{var}_distribution_{region}.png")


def plot_delta_ff_correlation_2d(
    x: np.ndarray,
    ff_sr: np.ndarray,
    ff_dr: np.ndarray,
    var_name: str,
    plot_dir: Path = None,
    bins_x: int = 50,
    bins_delta: int = 50,
    x_range: tuple[float, float] | None = None,
    delta_range: tuple[float, float] = (-6.0, 6.0),
) -> None:
    """Plot 2D histogram of Delta=log(FF_SR/FF_DR) vs one variable."""
    if plot_dir is None:
        plot_dir = SCRIPT_DIR

    x = np.asarray(x, dtype=np.float32)
    ff_sr = np.asarray(ff_sr, dtype=np.float32)
    ff_dr = np.asarray(ff_dr, dtype=np.float32)

    if not (len(x) == len(ff_sr) == len(ff_dr)):
        raise ValueError(
            f'Length mismatch for Delta plot ({var_name}): x={len(x)}, ff_sr={len(ff_sr)}, ff_dr={len(ff_dr)}.'
        )

    valid = np.isfinite(x) & np.isfinite(ff_sr) & np.isfinite(ff_dr) & (ff_sr > 0.0) & (ff_dr > 0.0)
    if not np.any(valid):
        logger.warning('No valid events for Delta 2D plot of %s; skipping.', var_name)
        return

    x_valid = x[valid]
    safe_ff_sr = np.clip(ff_sr[valid], 1e-8, None)
    safe_ff_dr = np.clip(ff_dr[valid], 1e-8, None)
    delta = np.log(safe_ff_sr / safe_ff_dr)
    finite_delta = np.isfinite(delta)
    x_valid = x_valid[finite_delta]
    delta = delta[finite_delta]

    if len(delta) == 0:
        logger.warning('No finite Delta values for %s; skipping 2D plot.', var_name)
        return

    pearson_r = float(np.corrcoef(x_valid, delta)[0, 1]) if len(delta) > 1 else np.nan
    spearman_r = float(scipy_stats.spearmanr(x_valid, delta).statistic) if len(delta) > 1 else np.nan

    if x_range is None:
        x_min = float(np.min(x_valid))
        x_max = float(np.max(x_valid))
        if x_max <= x_min:
            x_max = x_min + 1.0
        x_range = (x_min, x_max)

    x_bins = np.linspace(x_range[0], x_range[1], bins_x + 1)
    d_bins = np.linspace(delta_range[0], delta_range[1], bins_delta + 1)

    plt.figure(figsize=(8, 6))
    _, _, _, image = plt.hist2d(
        x_valid,
        delta,
        bins=[x_bins, d_bins],
        cmap='viridis',
        cmin=1,
        norm=matplotlib.colors.LogNorm(),
    )
    cbar = plt.colorbar(image)
    cbar.set_label('Event count (log scale)')

    plt.xlabel(var_name)
    plt.ylabel(r'$\Delta = \log(FF_{SR}/FF_{DR})$')
    #plt.title(f'Delta vs {var_name}')
    plt.grid(True, linestyle='--', alpha=0.3)
    ax = plt.gca()
    corr_text = (
        f'Pearson r = {pearson_r:.3f}\n'
        f'Spearman ρ = {spearman_r:.3f}'
        if np.isfinite(pearson_r) and np.isfinite(spearman_r)
        else 'Pearson r = n/a\nSpearman ρ = n/a'
    )
    ax.text(
        0.03,
        0.97,
        corr_text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'),
    )
    _apply_cms_plot_labels_outside(ax)
    plt.tight_layout()
    plt.savefig(plot_dir / f'delta_vs_{var_name}_2d.png')


# ──────────────────────────────────────────────
# Structured main helpers
# ──────────────────────────────────────────────

def _load_all_models(
    training_root: Path,
    config_path: Path,
    device: t.device,
    require_folded: bool = REQUIRE_OUT_OF_FOLD_MODELS,
) -> dict[str, t.nn.Module]:
    """Load all four NF models (AR-like, SR-like, AR, SR), auto-detecting FoldCombinedNF."""
    model_dirs = {
        'AR_like': training_root / 'Wjets' / 'DR'     / 'AR-like' / 'latest',
        'SR_like': training_root / 'Wjets' / 'DR'     / 'SR-like' / 'latest',
        'AR':      training_root / 'Wjets' / 'antiDR' / 'AR'      / 'latest',
        'SR':      training_root / 'Wjets' / 'antiDR' / 'SR'      / 'latest',
    }
    models = {}
    for name, chk_dir in model_dirs.items():
        cfg          = load_saved_model_config(chk_dir, config_path)
        models[name] = load_nf_model(
            dim=dim, cfg=cfg,
            checkpoint_path=chk_dir / 'model_checkpoint.pth',
            device=device,
            require_folded=require_folded,
        )
        logger.info('Loaded model %s (%s)', name, type(models[name]).__name__)
    return models


def _load_and_prepare_data(
    data_path: Path,
    masks_config: dict,
) -> dict[str, pd.DataFrame]:
    """Read feather, split MC/data, apply preselection and region masks."""
    data_complete = pd.read_feather(data_path)
    logger.info('Loaded %d total events from %s', len(data_complete), data_path)

    data_MC     = data_complete[data_complete['process'] == 1]
    data_events = data_complete[data_complete['process'] == 0]

    data_presel_MC     = _apply_config_mask(data_MC,     'mask_preselection_loose', masks_config)
    data_presel_events = _apply_config_mask(data_events, 'mask_preselection_loose', masks_config)

    return {
        'SR_MC':      _apply_config_mask(data_presel_MC,     'SR',      masks_config),
        'AR_MC':      _apply_config_mask(data_presel_MC,     'AR',      masks_config),
        'SR_like_MC': _apply_config_mask(data_presel_MC,     'SR_like', masks_config),
        'AR_like_MC': _apply_config_mask(data_presel_MC,     'AR_like', masks_config),
        'SR_events':  _apply_config_mask(data_presel_events, 'SR',      masks_config),
        'AR_events':  _apply_config_mask(data_presel_events, 'AR',      masks_config),
    }


def _build_input_tensors(
    data_dict: dict[str, pd.DataFrame],
    input_variables: list[str],
    device: t.device,
) -> dict[str, t.Tensor]:
    """Build input tensors [event_var, njets, feat1, ...] for AR_MC and AR_events."""
    for name in ['AR_MC', 'AR_events']:
        if 'event_var' not in data_dict[name].columns:
            raise KeyError(f"{name} dataframe is missing required column 'event_var'.")

    X_AR_MC = get_my_data(data_dict['AR_MC'], input_variables).to_torch(device=None).X.to(device)
    X_AR    = get_my_data_events(data_dict['AR_events'], input_variables).to_torch(device=None).X.to(device)
    _validate_event_var_tensor(X_AR_MC, 'AR_MC')
    _validate_event_var_tensor(X_AR, 'AR_events')
    return {'AR_MC': X_AR_MC, 'AR_events': X_AR}


def _evaluate_all_log_pdfs(
    models: dict[str, t.nn.Module],
    tensors: dict[str, t.Tensor],
) -> dict[str, dict]:
    """Evaluate log-PDFs for all (model, tensor) combinations.

    Returns: results[model_name][tensor_name] = {'log_pdf': ndarray, 'valid_mask': ndarray}
    """
    results = {}
    for model_name, model in models.items():
        if REQUIRE_OUT_OF_FOLD_MODELS and not isinstance(model, FoldCombinedNF):
            raise RuntimeError(
                f'{model_name}: model type {type(model).__name__} is not FoldCombinedNF. '
                'Out-of-fold fake-factor evaluation requires fold-combined models.'
            )
        results[model_name] = {}
        for tensor_name, X in tensors.items():
            lp, vm = evaluate_log_pdf(model, X)
            results[model_name][tensor_name] = {'log_pdf': lp, 'valid_mask': vm}
            logger.info(
                'Evaluated %s on %s: valid_fraction=%.4f',
                model_name, tensor_name, vm.mean(),
            )
    return results


def _compute_all_fake_factors(
    log_pdfs: dict,
    global_ffs: dict[str, float],
) -> dict[str, np.ndarray]:
    """Compute raw (unclipped) event-wise fake factors for data and MC."""
    def _lp(model: str, tensor: str) -> np.ndarray:
        return log_pdfs[model][tensor]['log_pdf']

    return {
        'ff_DR_events_raw': compute_fake_factors(_lp('AR_like', 'AR_events'), _lp('SR_like', 'AR_events'), global_ffs['DR'], clip_range=None),
        'ff_SR_events_raw': compute_fake_factors(_lp('AR',      'AR_events'), _lp('SR',      'AR_events'), global_ffs['SR'], clip_range=None),
        'ff_DR_MC_raw':     compute_fake_factors(_lp('AR_like', 'AR_MC'),     _lp('SR_like', 'AR_MC'),     global_ffs['DR'], clip_range=None),
        'ff_SR_MC_raw':     compute_fake_factors(_lp('AR',      'AR_MC'),     _lp('SR',      'AR_MC'),     global_ffs['SR'], clip_range=None),
    }


def main():
    masks_config = load_masks_config(MASKS_CONFIG_PATH)
    device = t.device('cuda:1' if t.cuda.is_available() else 'cpu')
    logger.info('Using device: %s', device)

    # --- models ---
    models = _load_all_models(TRAINING_ROOT, config_path, device, require_folded=REQUIRE_OUT_OF_FOLD_MODELS)

    # --- data ---
    data_dict       = _load_and_prepare_data(DATA_DIR / 'data_complete.feather', masks_config)
    input_variables = ['njets'] + list(variables)
    tensors         = _build_input_tensors(data_dict, input_variables, device)

    # --- log-PDFs ---
    log_pdfs = _evaluate_all_log_pdfs(models, tensors)

    # --- global fake factors ---
    global_ffs = {
        'DR': np.sum(data_dict['SR_like_MC']['weight']) / np.sum(data_dict['AR_like_MC']['weight']),
        'SR': np.sum(data_dict['SR_MC']['weight'])      / np.sum(data_dict['AR_MC']['weight']),
    }
    logger.info('Global FF  DR=%.6f  SR=%.6f', global_ffs['DR'], global_ffs['SR'])

    # --- per-event fake factors ---
    ffs      = _compute_all_fake_factors(log_pdfs, global_ffs)
    ff_DR_MC = np.clip(ffs['ff_DR_MC_raw'], 0, 10)
    ff_SR_MC = np.clip(ffs['ff_SR_MC_raw'], 0, 10)

    # --- save MC fake factors ---
    add_fake_factors_to_feather(
        full_df=data_dict['AR_MC'],
        selected_df=data_dict['AR_MC'],
        ff_dr=ffs['ff_DR_MC_raw'],
        ff_sr=ffs['ff_SR_MC_raw'],
        feather_path=FF_FACTORS_DIR / f'fake_factors_{resolved_tag}.feather',
    )

    # --- plots ---
    plot_ff_distributions(
        ffs['ff_SR_events_raw'], ffs['ff_DR_events_raw'],
        title='',
        plot_dir=PLOTS_DIR,
    )
    for var in variables:
        plt_control_plots(SR=data_dict['SR_MC'], AR=data_dict['AR_MC'], FF=ff_SR_MC, clip_value=15.0,
                          var=var, range=variable_ranges[var], region='DR MC', plot_dir=PLOTS_DIR)
        plt_control_plots(SR=data_dict['SR_MC'], AR=data_dict['AR_MC'], FF=ff_DR_MC, clip_value = 2.0,
                          var=var, range=variable_ranges[var], region='SR MC', plot_dir=PLOTS_DIR)

        # Delta uses AR_MC fake factors (same dataset and event order)
        if var == 'njets':
            plot_delta_ff_correlation_2d(
                x=data_dict['AR_MC'][var].to_numpy(dtype=np.float32),
                ff_sr=ffs['ff_SR_MC_raw'],
                ff_dr=ffs['ff_DR_MC_raw'],
                var_name=var,
                plot_dir=PLOTS_DIR,
                bins_x=10,
                bins_delta=50,
                x_range=variable_ranges[var])
        else:
            plot_delta_ff_correlation_2d(
                x=data_dict['AR_MC'][var].to_numpy(dtype=np.float32),
                ff_sr=ffs['ff_SR_MC_raw'],
                ff_dr=ffs['ff_DR_MC_raw'],
                var_name=var,
                plot_dir=PLOTS_DIR,
                bins_x=50,
                bins_delta=50,
                x_range=variable_ranges[var])
        



    for var in variables_correlation:
        # Delta uses AR_MC fake factors (same dataset and event order)
        plot_delta_ff_correlation_2d(
            x=data_dict['AR_MC'][var].to_numpy(dtype=np.float32),
            ff_sr=ffs['ff_SR_MC_raw'],
            ff_dr=ffs['ff_DR_MC_raw'],
            var_name=var,
            plot_dir=PLOTS_DIR / 'correlation',
            bins_x=50,
            bins_delta=50,
            x_range=variable_ranges[var]
        )
    

    

# ----------

if __name__ == '__main__':
    main()