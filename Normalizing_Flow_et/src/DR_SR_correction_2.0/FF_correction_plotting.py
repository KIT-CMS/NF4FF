import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import torch as t
import torch.nn as nn
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / 'configs'
DATA_DIR = WORKSPACE_ROOT / 'data'
MASKS_CONFIG_PATH = CONFIG_DIR / 'masks_MC.yaml'

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from classes.NeuralNetworks import ConditionalFlow1D
from classes.Plotting import CMS_CHANNEL_TITLE, CMS_LABEL, CMS_LUMI_TITLE, CMS_NJETS_TITLE

logger = logging.getLogger(__name__)

FF_FLOW_RESULTS_ROOT = SCRIPT_DIR / 'FF_flow_results'

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


def _apply_cms_plot_labels(ax) -> None:
    CMS_CHANNEL_TITLE([ax])
    CMS_LUMI_TITLE([ax])
    CMS_LABEL([ax])
    CMS_NJETS_TITLE([ax], title=r"$N_{jets} \geq 0$")


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


def _load_ar_mc_like_ff_plotting(
    data_path: str | Path,
    masks_config: dict[str, list[str]],
) -> pd.DataFrame:
    """Match FF_plotting data-loading sequence and return AR_MC sample."""
    data_path = Path(data_path)
    data_complete = pd.read_feather(data_path)
    logger.info('Loaded %d total events from %s', len(data_complete), data_path)

    data_mc = data_complete[data_complete['process'] == 1]
    data_presel_mc = _apply_config_mask(data_mc, 'mask_preselection_loose', masks_config)
    data_ar_mc = _apply_config_mask(data_presel_mc, 'AR', masks_config)
    logger.info('AR_MC sample after FF_plotting-style masks: %d events', len(data_ar_mc))
    return data_ar_mc


def resolve_training_name(variables: list[str]) -> str:
    tail = variables[4:]
    tag = '_'.join(tail) if tail else 'none'
    return f'training_vars{len(variables)}_{tag}'


def resolve_training_name_correction(variables: list[str]) -> str:
    tail = variables
    tag = '_'.join(tail) if tail else 'none'
    return f'training_vars{len(variables)}_{tag}'


class FoldCombinedConditionalFlow1D(nn.Module):
    """Route events using event_var: 0 -> even_model, 1 -> odd_model."""

    def __init__(self, even_model: ConditionalFlow1D, odd_model: ConditionalFlow1D):
        super().__init__()
        self.even_model = even_model
        self.odd_model = odd_model

    def log_prob(self, y: t.Tensor, cond_with_event: t.Tensor) -> t.Tensor:
        event_var = cond_with_event[:, 0].long()
        cond = cond_with_event[:, 1:]
        y_flat = y.squeeze(-1) if y.ndim > 1 else y
        out = t.empty_like(y_flat)

        even_mask = event_var == 0
        odd_mask = ~even_mask
        if even_mask.any():
            out[even_mask] = self.even_model.log_prob(y_flat[even_mask], cond[even_mask])
        if odd_mask.any():
            out[odd_mask] = self.odd_model.log_prob(y_flat[odd_mask], cond[odd_mask])
        return out

    def sample(self, cond_with_event: t.Tensor, n_samples: int = 1) -> t.Tensor:
        event_var = cond_with_event[:, 0].long()
        cond = cond_with_event[:, 1:]
        out = t.empty((n_samples, len(cond)), device=cond.device, dtype=cond.dtype)

        even_mask = event_var == 0
        odd_mask = ~even_mask
        if even_mask.any():
            out[:, even_mask] = self.even_model.sample(cond[even_mask], n_samples=n_samples)
        if odd_mask.any():
            out[:, odd_mask] = self.odd_model.sample(cond[odd_mask], n_samples=n_samples)
        return out


def load_ff_correction_flow_results(
    variables_mc: list[str] | None = None,
    variables_correction: list[str] | None = None,
    checkpoint_path: str | Path | None = None,
    results_root: str | Path = FF_FLOW_RESULTS_ROOT,
    device: t.device | None = None,
) -> dict:
    """Load correction-flow checkpoint/scalers and build the folded model object.

    Provide either checkpoint_path directly or both variables_mc and variables_correction.
    """
    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    if checkpoint_path is None:
        if variables_mc is None or variables_correction is None:
            raise ValueError('Provide checkpoint_path or (variables_mc and variables_correction).')
        mc_tag = resolve_training_name(variables_mc)
        corr_tag = resolve_training_name_correction(variables_correction)
        checkpoint_path = Path(results_root) / mc_tag / corr_tag / 'FF_SR' / 'model_checkpoint.pth'
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    checkpoint = t.load(checkpoint_path, map_location=device)
    scaler_meta_path = checkpoint_path.parent / 'scaler_meta.yaml'
    scaler_meta = {}
    if scaler_meta_path.exists():
        with open(scaler_meta_path, 'r') as fh:
            scaler_meta = yaml.safe_load(fh) or {}

    schema = checkpoint.get('schema', scaler_meta.get('schema', 'single_correction_flow_v1'))
    cond_dim = int(checkpoint.get('cond_dim', scaler_meta.get('cond_dim')))
    if cond_dim <= 0:
        raise ValueError('Could not determine cond_dim from checkpoint/scaler metadata.')

    if schema in {
        'fold_combined_correction_flow_ffsr_log_v1',
        'fold_combined_correction_flow_ffsr_v1',
        'fold_combined_correction_flow_v1',
    }:
        even_model = ConditionalFlow1D(cond_dim).to(device)
        odd_model = ConditionalFlow1D(cond_dim).to(device)
        even_model.load_state_dict(checkpoint['even_model_state_dict'])
        odd_model.load_state_dict(checkpoint['odd_model_state_dict'])
        even_model.eval()
        odd_model.eval()
        model = FoldCombinedConditionalFlow1D(even_model=even_model, odd_model=odd_model).to(device)
        model.eval()
    elif 'model_state_dict' in checkpoint:
        model = ConditionalFlow1D(cond_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        even_model = None
        odd_model = None
    else:
        raise ValueError(f'Unsupported correction checkpoint schema: {schema}')

    logger.info('Loaded correction flow from %s (schema=%s)', checkpoint_path, schema)
    return {
        'model': model,
        'even_model': even_model,
        'odd_model': odd_model,
        'checkpoint': checkpoint,
        'scaler_meta': scaler_meta,
        'checkpoint_path': checkpoint_path,
        'schema': schema,
        'cond_dim': cond_dim,
        'device': device,
    }


def calculate_ff_sr_with_correction_flow(
    ff_dr: np.ndarray,
    x: np.ndarray,
    loaded_flow: dict,
    event_var: np.ndarray | None = None,
    n_samples: int = 1,
    reduction: str = 'median',
) -> np.ndarray:
    """Predict FF_SR from FF_DR and X (correction variables), using loaded correction flow."""
    ff_dr = np.asarray(ff_dr, dtype=np.float32).reshape(-1)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if len(ff_dr) != len(x):
        raise ValueError(f'Length mismatch: len(ff_dr)={len(ff_dr)} vs len(x)={len(x)}')

    cond_dim = int(loaded_flow['cond_dim'])
    expected_x_dim = cond_dim - 1
    x_vars = x

    if event_var is None and x.shape[1] == expected_x_dim + 1:
        maybe_event = x[:, 0]
        if np.all(np.isin(maybe_event.astype(int), [0, 1])):
            event_var = maybe_event.astype(np.int64)
            x_vars = x[:, 1:]

    if x_vars.shape[1] != expected_x_dim:
        raise ValueError(
            f'X has wrong feature dimension: got {x_vars.shape[1]}, expected {expected_x_dim}.'
        )

    valid_mask = np.isfinite(ff_dr) & (ff_dr > 0) & np.all(np.isfinite(x_vars), axis=1)

    if loaded_flow['schema'] in {
        'fold_combined_correction_flow_ffsr_log_v1',
        'fold_combined_correction_flow_ffsr_v1',
        'fold_combined_correction_flow_v1',
    }:
        if event_var is None:
            raise ValueError('event_var is required for fold-combined correction flow schemas.')
        event_var_np = np.asarray(event_var, dtype=np.int64).reshape(-1)
        if len(event_var_np) != len(ff_dr):
            raise ValueError('Length mismatch between event_var and ff_dr.')
        valid_mask &= np.isin(event_var_np, [0, 1])
    else:
        event_var_np = None

    ff_sr_pred = np.full(len(ff_dr), np.nan, dtype=np.float32)
    n_valid = int(valid_mask.sum())
    n_total = len(valid_mask)
    n_invalid = n_total - n_valid
    if n_valid == 0:
        logger.warning('No valid rows for correction-flow prediction (all outputs set to NaN).')
        return ff_sr_pred
    if n_invalid > 0:
        logger.warning('Skipping %d/%d invalid rows for correction-flow prediction.', n_invalid, n_total)

    ff_dr_valid = ff_dr[valid_mask]
    x_valid = x_vars[valid_mask]

    cond_np = np.column_stack([np.log(ff_dr_valid), x_valid]).astype(np.float32)
    cond = t.tensor(cond_np, dtype=t.float32, device=loaded_flow['device'])

    with t.no_grad():
        if loaded_flow['schema'] in {
            'fold_combined_correction_flow_ffsr_log_v1',
            'fold_combined_correction_flow_ffsr_v1',
            'fold_combined_correction_flow_v1',
        }:
            cond_with_event = np.column_stack([event_var_np[valid_mask], cond_np]).astype(np.float32)
            cond_tensor = t.tensor(cond_with_event, dtype=t.float32, device=loaded_flow['device'])
            ff_sr_samples = loaded_flow['model'].sample(cond_tensor, n_samples=n_samples)
        else:
            ff_sr_samples = loaded_flow['model'].sample(cond, n_samples=n_samples)

    ff_sr_samples_np = ff_sr_samples.detach().cpu().numpy()
    scaler_meta = loaded_flow.get('scaler_meta', {}) or {}
    target_definition = scaler_meta.get('target_definition', '')

    if target_definition == 'log(FF_SR/FF_DR)':
        delta_samples = ff_sr_samples_np
        ratio_samples = np.exp(delta_samples)
        if reduction == 'mean':
            ff_sr_pred_valid = ratio_samples.mean(axis=0) * ff_dr_valid
        elif reduction == 'median':
            ff_sr_pred_valid = np.median(ratio_samples, axis=0) * ff_dr_valid
        else:
            raise ValueError("reduction must be one of {'mean', 'median'}")
    else:
        if reduction == 'mean':
            ff_sr_pred_valid = ff_sr_samples_np.mean(axis=0)
        elif reduction == 'median':
            ff_sr_pred_valid = np.median(ff_sr_samples_np, axis=0)
        else:
            raise ValueError("reduction must be one of {'mean', 'median'}")


    ff_sr_pred[valid_mask] = ff_sr_pred_valid.astype(np.float32)
    return ff_sr_pred.astype(np.float32)


# ============================================
# CORRECTION FLOW VALIDATION UTILITIES
# ============================================


def compute_delta(ff_sr: np.ndarray, ff_dr: np.ndarray) -> np.ndarray:
    """
    Compute Δ = log(FF_SR / FF_DR)
    """
    ff_sr = np.asarray(ff_sr, dtype=np.float32)
    ff_dr = np.asarray(ff_dr, dtype=np.float32)

    delta = np.full_like(ff_sr, np.nan, dtype=np.float32)
    mask = (ff_sr > 0) & (ff_dr > 0)

    safe_ff_dr = np.clip(ff_dr[mask], 1e-8, None)
    safe_ff_sr = np.clip(ff_sr[mask], 1e-8, None)

    delta[mask] = np.log(safe_ff_sr / safe_ff_dr)


    return delta


# ============================================
# 1) DELTA DISTRIBUTION (MAIN VALIDATION)
# ============================================

def plot_delta_distribution(
    ff_dr: np.ndarray,
    ff_sr_true: np.ndarray,
    ff_sr_pred: np.ndarray,
    bins: int = 80,
    out_path: Path | None = None,
):
    delta_true = compute_delta(ff_sr_true, ff_dr)
    delta_pred = compute_delta(ff_sr_pred, ff_dr)

    mask = np.isfinite(delta_true) & np.isfinite(delta_pred)

    delta_true = delta_true[mask]
    delta_pred = delta_pred[mask]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(delta_true, bins=bins, density=True,
            histtype='step', linewidth=2,
            label='Δ true', color='tab:blue')

    ax.hist(delta_pred, bins=bins, density=True,
            histtype='step', linewidth=2,
            label='Δ predicted', color='tab:orange')

    ax.set_xlabel(r'$\Delta = \log(FF_{SR}/FF_{DR})$')
    ax.set_ylabel('Density')
    ax.set_title('Correction Δ Distribution')
    ax.grid(alpha=0.3)
    ax.legend()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ============================================
# 2) YIELD CLOSURE (MOST IMPORTANT)
# ============================================

def print_yield_closure(ff_sr_true: np.ndarray, ff_sr_pred: np.ndarray):
    mask = np.isfinite(ff_sr_true) & np.isfinite(ff_sr_pred)

    if not mask.any():
        print("No valid events.")
        return

    true_yield = np.sum(ff_sr_true[mask])
    pred_yield = np.sum(ff_sr_pred[mask])

    ratio = pred_yield / true_yield if true_yield > 0 else np.nan

    print("\n===== YIELD CLOSURE =====")
    print(f"True yield         : {true_yield:.6e}")
    print(f"Predicted yield    : {pred_yield:.6e}")
    print(f"Pred / True ratio  : {ratio:.4f}")
    print("=========================\n")


# ============================================
# 3) DELTA VS VARIABLE (CHECK LEARNED STRUCTURE)
# ============================================

VARIABLE_RANGES: dict[str, tuple[float, float]] = {
    'pt_1':              (30,  100),
    'pt_2':              (30,  100),
    'm_vis':             (0,   250),
    'deltaR_ditaupair':  (0.3, 5.0),
    'pt_vis':            (0,   200),
    'met':               (0,   150),
    'pt_tt':             (0,   200),
    'm_fastmtt':         (0,   250),
    'mt_tot':            (0,   200),
    'mt_1':              (0,    70),
    'mt_2':              (0,   150),
    'njets':             (0,    10),
}

DELTA_RANGE: tuple[float, float] = (-10.0, 10.0)


def plot_delta_vs_variable(
    variable: np.ndarray,
    ff_dr: np.ndarray,
    ff_sr_true: np.ndarray,
    ff_sr_pred: np.ndarray,
    var_name: str,
    n_bins: int = 20,
    x_range: tuple[float, float] | None = None,
    delta_range: tuple[float, float] = DELTA_RANGE,
    out_path: Path | None = None,
):
    delta_true = compute_delta(ff_sr_true, ff_dr)
    delta_pred = compute_delta(ff_sr_pred, ff_dr)

    mask = (
        np.isfinite(variable)
        & np.isfinite(delta_true)
        & np.isfinite(delta_pred)
    )

    var = variable[mask]
    dt = delta_true[mask]
    dp = delta_pred[mask]

    if len(var) == 0:
        logger.warning('No finite rows for delta-vs-variable plot: %s', var_name)
        return

    def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Return (Pearson r, Spearman ρ)."""
        if len(x) < 2:
            return float('nan'), float('nan')
        if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
            return float('nan'), float('nan')
        pearson = float(np.corrcoef(x, y)[0, 1])
        spearman = float(scipy_stats.spearmanr(x, y).statistic)
        return pearson, spearman

    pearson_true, spearman_true = _safe_corr(var, dt)
    pearson_pred, spearman_pred = _safe_corr(var, dp)

    fig, (ax_true, ax_pred) = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)

    x_min, x_max = x_range if x_range is not None else (float(np.min(var)), float(np.max(var)))
    d_min, d_max = delta_range

    h_true = ax_true.hist2d(
        var,
        dt,
        bins=[n_bins, n_bins],
        range=[[x_min, x_max], [d_min, d_max]],
        cmap='viridis',
        norm=matplotlib.colors.LogNorm(),
        cmin=1,
    )
    h_pred = ax_pred.hist2d(
        var,
        dp,
        bins=[n_bins, n_bins],
        range=[[x_min, x_max], [d_min, d_max]],
        cmap='viridis',
        norm=matplotlib.colors.LogNorm(),
        cmin=1,
    )

    ax_true.set_title(f'Δ true vs {var_name}')
    ax_pred.set_title(f'Δ pred vs {var_name}')
    ax_true.set_xlabel(var_name)
    ax_pred.set_xlabel(var_name)
    ax_true.set_ylabel(r'$\Delta = \log(FF_{SR}/FF_{DR})$')

    ax_true.text(
        0.03,
        0.97,
        (
            f'Pearson r = {pearson_true:.3f}\nSpearman ρ = {spearman_true:.3f}'
            if np.isfinite(pearson_true) and np.isfinite(spearman_true)
            else 'Pearson r = n/a\nSpearman ρ = n/a'
        ),
        transform=ax_true.transAxes,
        ha='left',
        va='top',
        bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
    )
    ax_pred.text(
        0.03,
        0.97,
        (
            f'Pearson r = {pearson_pred:.3f}\nSpearman ρ = {spearman_pred:.3f}'
            if np.isfinite(pearson_pred) and np.isfinite(spearman_pred)
            else 'Pearson r = n/a\nSpearman ρ = n/a'
        ),
        transform=ax_pred.transAxes,
        ha='left',
        va='top',
        bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
    )

    cbar_true = fig.colorbar(h_true[3], ax=ax_true)
    cbar_true.set_label('Event count (log)')
    cbar_pred = fig.colorbar(h_pred[3], ax=ax_pred)
    cbar_pred.set_label('Event count (log)')

    ax_true.grid(alpha=0.2)
    ax_pred.grid(alpha=0.2)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ============================================
# 4) MAIN VALIDATION DRIVER (DROP-IN REPLACEMENT)
# ============================================

def validate_correction_flow(
    ff_dr: np.ndarray,
    ff_sr_true: np.ndarray,
    ff_sr_pred: np.ndarray,
    variables: np.ndarray | None = None,
    variable_names: list[str] | None = None,
    out_dir: str | Path | None = None,
):
    """
    Main validation function to replace your current histogram comparison.
    """

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Delta distribution
    plot_delta_distribution(
        ff_dr,
        ff_sr_true,
        ff_sr_pred,
        out_path=None if out_dir is None else out_dir / "delta_distribution.png"
    )

    # --- 2. Yield closure
    print_yield_closure(ff_sr_true, ff_sr_pred)

    # --- 3. Delta vs variables
    if variables is not None and variable_names is not None:
        for i, name in enumerate(variable_names):
            plot_delta_vs_variable(
                variables[:, i],
                ff_dr,
                ff_sr_true,
                ff_sr_pred,
                var_name=name,
                x_range=VARIABLE_RANGES.get(name),
                delta_range=DELTA_RANGE,
                out_path=None if out_dir is None else out_dir / f"delta_vs_{name}.png"
            )

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    with open(CONFIG_DIR / 'training_variables.yaml', 'r') as fh:
        training_variables_cfg = yaml.safe_load(fh)
    variables_mc = training_variables_cfg['variables_MC']
    variables_correction = training_variables_cfg['variables_correction']

    # Load data in the same style as FF_plotting.py
    masks_config = load_masks_config(MASKS_CONFIG_PATH)
    ar_mc_df = _load_ar_mc_like_ff_plotting(DATA_DIR / 'data_complete.feather', masks_config)

    # Load FF columns in the same style as FF_correction_flow.py
    resolved_tag = resolve_training_name(variables_mc)
    fake_factors_path = SCRIPT_DIR / 'Fake_Factors' / f'fake_factors_{resolved_tag}.feather'
    if not fake_factors_path.exists():
        raise FileNotFoundError(
            f'Fake factors file not found: {fake_factors_path}\nRun FF_plotting.py first.'
        )

    ff_df = pd.read_feather(fake_factors_path)
    logger.info('Loaded %d fake-factor rows from %s', len(ff_df), fake_factors_path)

    if len(ar_mc_df) != len(ff_df):
        logger.warning(
            'AR_MC length (%d) differs from fake-factor length (%d). Plotting uses fake-factor rows.',
            len(ar_mc_df),
            len(ff_df),
        )

    loaded_flow = load_ff_correction_flow_results(
        variables_mc=variables_mc,
        variables_correction=variables_correction,
    )

    required_columns = ['FF_DR', 'FF_SR', 'event_var'] + list(variables_correction)
    valid_mask = np.ones(len(ff_df), dtype=bool)
    for col in required_columns:
        valid_mask &= np.isfinite(ff_df[col].to_numpy(dtype=np.float32))
    valid_mask &= ff_df['event_var'].isin([0, 1]).to_numpy(dtype=bool)
    valid_mask &= (ff_df['FF_DR'].to_numpy(dtype=np.float32) > 0.0)
    valid_mask &= (ff_df['FF_SR'].to_numpy(dtype=np.float32) > 0.0)

    n_before = len(ff_df)
    ff_df = ff_df[valid_mask].reset_index(drop=True)
    n_after = len(ff_df)
    if n_after != n_before:
        logger.info('Applied training-compatible validation mask: kept %d/%d events', n_after, n_before)

    ff_sr_pred = calculate_ff_sr_with_correction_flow(
        ff_dr=ff_df['FF_DR'].to_numpy(dtype=np.float32),
        x=ff_df[variables_correction].to_numpy(dtype=np.float32),
        loaded_flow=loaded_flow,
        event_var=ff_df['event_var'].to_numpy(dtype=np.int64),
        n_samples=20,
        reduction='mean',
    )

    validate_correction_flow(
        ff_dr=ff_df['FF_DR'].to_numpy(dtype=np.float32),
        ff_sr_true=ff_df['FF_SR'].to_numpy(dtype=np.float32),
        ff_sr_pred=ff_sr_pred,
        variables=ff_df[variables_correction].to_numpy(dtype=np.float32),
        variable_names=variables_correction,
        out_dir=SCRIPT_DIR / 'plots' / 'correction_validation',
    )

    logger.info('Finished correction-flow fold-wise comparison plots.')


if __name__ == '__main__':
    main()