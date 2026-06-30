import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
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

    cond_np = np.column_stack([ff_dr_valid, x_valid]).astype(np.float32)
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
    if reduction == 'mean':
        ff_sr_pred_valid = ff_sr_samples_np.mean(axis=0)
    elif reduction == 'median':
        ff_sr_pred_valid = np.median(ff_sr_samples_np, axis=0)
    else:
        raise ValueError("reduction must be one of {'mean', 'median'}")

    use_log_target = bool(loaded_flow.get('scaler_meta', {}).get('use_log_transform', False))
    if use_log_target:
        ff_sr_pred_valid = np.exp(np.clip(ff_sr_pred_valid, -80.0, 80.0))

    ff_sr_pred[valid_mask] = ff_sr_pred_valid.astype(np.float32)
    return ff_sr_pred.astype(np.float32)


def plot_foldwise_ff_dr_ff_sr_true_vs_pred(
    ff_dr: np.ndarray,
    ff_sr_true: np.ndarray,
    x: np.ndarray,
    event_var: np.ndarray,
    loaded_flow: dict,
    bins: int = 80,
    density: bool = True,
    n_samples: int = 1,
    x_range: tuple[float, float] = (1e-9, 1e-5),
    out_dir: str | Path | None = None,
    filename_prefix: str = 'ff_correction_fold',
) -> dict[str, dict[str, np.ndarray]]:
    """For each fold, create FF comparison plots analogous to FF_calculation.py.

    Produces per fold:
    - FF_DR vs FF_SR(true) vs FF_SR(pred) comparison histogram
    - FF_SR(pred) / FF_SR(true) ratio panel histogram
    """
    if loaded_flow.get('schema') not in {
        'fold_combined_correction_flow_ffsr_log_v1',
        'fold_combined_correction_flow_ffsr_v1',
        'fold_combined_correction_flow_v1',
    }:
        raise ValueError('Per-fold plotting requires a fold-combined correction flow model.')

    ff_dr = np.asarray(ff_dr, dtype=np.float32).reshape(-1)
    ff_sr_true = np.asarray(ff_sr_true, dtype=np.float32).reshape(-1)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    event_var = np.asarray(event_var, dtype=np.int64).reshape(-1)

    n = len(ff_dr)
    if not (len(ff_sr_true) == len(x) == len(event_var) == n):
        raise ValueError('Input length mismatch among ff_dr, ff_sr_true, x, and event_var.')

    cond_dim = int(loaded_flow['cond_dim'])
    expected_x_dim = cond_dim - 1
    if x.shape[1] != expected_x_dim:
        raise ValueError(f'X has wrong feature dimension: got {x.shape[1]}, expected {expected_x_dim}.')

    x_min_global = float(x_range[0])
    x_max_global = float(x_range[1])
    if not np.isfinite(x_min_global) or not np.isfinite(x_max_global):
        raise ValueError(f'x_range must be finite, got {x_range}.')
    if x_min_global <= 0.0 or x_max_global <= x_min_global:
        raise ValueError(f'x_range must satisfy 0 < min < max for log-scale axes, got {x_range}.')

    fold_map = {
        'even_events_eval_even_model': 0,
        'odd_events_eval_odd_model': 1,
    }
    results: dict[str, dict[str, np.ndarray]] = {}

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for fold_name, fold_value in fold_map.items():
        mask = event_var == fold_value
        if not mask.any():
            logger.warning('No events found for fold %s.', fold_name)
            continue

        ff_dr_fold = ff_dr[mask]
        ff_sr_true_fold = ff_sr_true[mask]
        x_fold = x[mask]
        ev_fold = event_var[mask]

        # For this fold, true FF_SR values were not used to train the corresponding sub-model.
        ff_sr_pred_fold = calculate_ff_sr_with_correction_flow(
            ff_dr=ff_dr_fold,
            x=x_fold,
            loaded_flow=loaded_flow,
            event_var=ev_fold,
            n_samples=n_samples,
            reduction='median',
        )

        valid_plot = (
            np.isfinite(ff_dr_fold)
            & (ff_dr_fold > 0)
            & np.isfinite(ff_sr_true_fold)
            & (ff_sr_true_fold > 0)
            & np.isfinite(ff_sr_pred_fold)
            & (ff_sr_pred_fold > 0)
        )
        if not valid_plot.any():
            logger.warning('No finite rows to plot for fold %s.', fold_name)
            continue

        ff_dr_plot = ff_dr_fold[valid_plot]
        ff_sr_true_plot = ff_sr_true_fold[valid_plot]
        ff_sr_pred_plot = ff_sr_pred_fold[valid_plot]

        in_range = (
            (ff_dr_plot >= x_min_global) & (ff_dr_plot <= x_max_global)
            & (ff_sr_true_plot >= x_min_global) & (ff_sr_true_plot <= x_max_global)
            & (ff_sr_pred_plot >= x_min_global) & (ff_sr_pred_plot <= x_max_global)
        )
        if not in_range.any():
            logger.warning(
                'No rows within x_range %s for fold %s; skipping.',
                x_range,
                fold_name,
            )
            continue

        ff_dr_plot = ff_dr_plot[in_range]
        ff_sr_true_plot = ff_sr_true_plot[in_range]
        ff_sr_pred_plot = ff_sr_pred_plot[in_range]

        x_min = x_min_global
        x_max = x_max_global

        bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), bins + 1)

        # 1) Three-way comparison: FF_SR(true), FF_DR, FF_SR(pred)
        fig_cmp, ax_cmp = plt.subplots(figsize=(8.5, 5.5))
        ax_cmp.hist(
            ff_sr_true_plot,
            bins=bin_edges,
            density=density,
            histtype='step',
            linewidth=2.0,
            label='FF_SR true (held-out)',
            color='tab:blue',
        )
        ax_cmp.hist(
            ff_dr_plot,
            bins=bin_edges,
            density=density,
            histtype='step',
            linewidth=2.0,
            label='FF_DR',
            color='tab:red',
        )
        ax_cmp.hist(
            ff_sr_pred_plot,
            bins=bin_edges,
            density=density,
            histtype='step',
            linewidth=2.0,
            label='FF_SR predicted (flow)',
            color='tab:green',
        )
        ax_cmp.set_title(f'Fake Factor Comparison (Fold: {fold_name})')
        ax_cmp.set_xlabel('Fake Factor')
        ax_cmp.set_ylabel('Density' if density else 'Count')
        ax_cmp.set_xscale('log')
        ax_cmp.set_xlim(left=x_min, right=x_max)
        ax_cmp.grid(alpha=0.25)
        ax_cmp.legend(loc='best')
        _apply_cms_plot_labels(ax_cmp)
        fig_cmp.tight_layout()

        # 2) FF_SR true vs FF_SR predicted with ratio panel (pred/true)
        counts_true, _ = np.histogram(ff_sr_true_plot, bins=bin_edges)
        counts_pred, _ = np.histogram(ff_sr_pred_plot, bins=bin_edges)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(counts_true > 0, counts_pred / counts_true, np.nan)

        fig_ratio, (ax_main, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(8.5, 7.0),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True,
        )
        ax_main.step(bin_edges[:-1], counts_true, where='post', color='tab:blue', label='FF_SR true (held-out)')
        ax_main.step(bin_edges[:-1], counts_pred, where='post', color='tab:orange', label='FF_SR predicted (flow)')
        ax_main.set_ylabel('Events')
        ax_main.set_xscale('log')
        ax_main.set_title(f'FF_SR: true vs flow prediction (Fold: {fold_name})')
        ax_main.grid(alpha=0.25)
        ax_main.legend(loc='best')
        _apply_cms_plot_labels(ax_main)

        ax_ratio.axhline(1.0, color='black', linewidth=0.8, linestyle='--')
        ax_ratio.step(bin_edges[:-1], ratio, where='post', color='tab:orange')
        ax_ratio.set_xlabel('Fake Factor')
        ax_ratio.set_ylabel('Flow/True')
        ax_ratio.set_ylim(0.0, 2.0)
        ax_ratio.set_xscale('log')
        ax_ratio.grid(alpha=0.25)
        ax_ratio.set_xlim(left=x_min, right=x_max)
        fig_ratio.tight_layout()

        if out_dir is not None:
            out_path_cmp = Path(out_dir) / f'{filename_prefix}_{fold_name}_comparison.png'
            out_path_ratio = Path(out_dir) / f'{filename_prefix}_{fold_name}_ffsr_ratio.png'
            fig_cmp.savefig(out_path_cmp, dpi=160)
            fig_ratio.savefig(out_path_ratio, dpi=160)
            logger.info('Saved fold comparison plot to %s', out_path_cmp)
            logger.info('Saved fold FF_SR ratio plot to %s', out_path_ratio)
            plt.close(fig_cmp)
            plt.close(fig_ratio)

        results[fold_name] = {
            'ff_dr': ff_dr_fold,
            'ff_sr_true': ff_sr_true_fold,
            'ff_sr_pred': ff_sr_pred_fold,
        }

    return results


def run_foldwise_ff_correction_plots_from_fake_factors(
    variables_mc: list[str],
    variables_correction: list[str],
    fake_factors_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    n_samples: int = 20,
    bins: int = 80,
    density: bool = True,
    x_range: tuple[float, float] = (1e-9, 1e-5),
) -> dict[str, dict[str, np.ndarray]]:
    """Convenience wrapper: load model + fake-factor file and create fold-wise plots.

    It loads:
    - correction flow from FF_flow_results
    - fake factors feather file from Fake_Factors
    and then produces per-fold histograms.
    """
    if fake_factors_path is None:
        fake_tag = resolve_training_name(variables_mc)
        fake_factors_path = SCRIPT_DIR / 'Fake_Factors' / f'fake_factors_{fake_tag}.feather'
    else:
        fake_factors_path = Path(fake_factors_path)

    if not fake_factors_path.exists():
        raise FileNotFoundError(f'Fake factors file not found: {fake_factors_path}')

    required_columns = ['FF_DR', 'FF_SR', 'event_var'] + list(variables_correction)
    df = pd.read_feather(fake_factors_path)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f'Missing required columns in {fake_factors_path}: {missing}')

    loaded = load_ff_correction_flow_results(
        variables_mc=variables_mc,
        variables_correction=variables_correction,
        checkpoint_path=checkpoint_path,
    )

    if out_dir is None:
        out_dir = SCRIPT_DIR / 'plots' / 'ff_correction_foldwise'

    results = plot_foldwise_ff_dr_ff_sr_true_vs_pred(
        ff_dr=df['FF_DR'].to_numpy(dtype=np.float32),
        ff_sr_true=df['FF_SR'].to_numpy(dtype=np.float32),
        x=df[variables_correction].to_numpy(dtype=np.float32),
        event_var=df['event_var'].to_numpy(dtype=np.int64),
        loaded_flow=loaded,
        bins=bins,
        density=density,
        n_samples=n_samples,
        x_range=x_range,
        out_dir=out_dir,
    )

    logger.info('Completed fold-wise correction plots using fake factors from %s', fake_factors_path)
    return results


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

    out_dir = SCRIPT_DIR / 'plots' / 'ff_correction_foldwise'
    run_foldwise_ff_correction_plots_from_fake_factors(
        variables_mc=variables_mc,
        variables_correction=variables_correction,
        fake_factors_path=fake_factors_path,
        out_dir=out_dir,
        n_samples=20,
        bins=80,
        density=True,
        x_range=(1e-9, 1e-5),
    )

    logger.info('Finished correction-flow fold-wise comparison plots.')


if __name__ == '__main__':
    main()