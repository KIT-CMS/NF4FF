import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / 'configs'
DATA_DIR = WORKSPACE_ROOT / 'data'
MASKS_CONFIG_PATH = CONFIG_DIR / 'masks_MC.yaml'

if str(SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPT_DIR))

from classes.NeuralNetworks import ConditionalRealNVP
from classes.Collection import load_model_config, ModelConfig, load_conditional_flow
from CustomLogging import setup_logging


logger = setup_logging(logger=logging.getLogger(__name__))

mode_dir = 'conditional_njets_input'
config_path = CONFIG_DIR / 'config_NF.yaml'
PLOT_ROOT = SCRIPT_DIR / 'FF_comparison_results'
PLOT_ROOT.mkdir(parents=True, exist_ok=True)


def resolve_training_name(variables: list[str]) -> str:
	tail = variables[4:]
	tag = '_'.join(tail) if tail else 'none'
	return f"training_vars{len(variables)}_{tag}"


with open(CONFIG_DIR / 'training_variables.yaml', 'r') as f:
	training_variables_cfg = yaml.safe_load(f)
	variables = training_variables_cfg['variables_MC']


def get_my_data_events(df: pd.DataFrame, training_var: list[str]) -> np.ndarray:
	return df[training_var].to_numpy(dtype=np.float32)


def load_saved_model_config(checkpoint_dir: str | Path, fallback_path: str | Path) -> ModelConfig:
	saved_config_path = Path(checkpoint_dir).parent / 'config.yaml'
	if saved_config_path.exists():
		with open(saved_config_path, 'r') as handle:
			raw = yaml.unsafe_load(handle)

		if isinstance(raw, ModelConfig):
			return raw

		values = vars(raw) if hasattr(raw, '__dict__') else raw
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
	config_path_local = Path(path)
	if not config_path_local.exists():
		raise FileNotFoundError(f'Mask configuration not found: {config_path_local}')

	with open(config_path_local, 'r') as handle:
		raw = yaml.safe_load(handle) or {}

	masks = raw.get('masks', raw)
	if not isinstance(masks, dict):
		raise ValueError(f'Invalid masks configuration in {config_path_local}: expected a mapping at top level.')

	normalized: dict[str, list[str]] = {}
	for name, expressions in masks.items():
		if isinstance(expressions, str):
			normalized[name] = [expressions]
		elif isinstance(expressions, list) and all(isinstance(expr, str) for expr in expressions):
			normalized[name] = expressions
		else:
			raise TypeError(
				f"Mask '{name}' in {config_path_local} must be a string or list[str], got {type(expressions).__name__}."
			)

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


@t.no_grad()
def evaluate_log_pdf(model: ConditionalRealNVP, X: t.Tensor) -> tuple[np.ndarray, np.ndarray]:
	cond_dim = int(getattr(model, 'cond_dim', 0))
	x_features = X[:, cond_dim:]
	x_preprocessed, _, valid_mask = model.apply_preprocessing(x_features)
	_ = model.apply_scaler(x_preprocessed)

	log_pdf = model(X)
	log_pdf = t.nan_to_num(log_pdf, nan=-700.0, neginf=-700.0, posinf=700.0)
	return log_pdf.detach().cpu().numpy(), valid_mask.detach().cpu().numpy()


def compute_fake_factors(
	log_pdf_ar: np.ndarray,
	log_pdf_sr: np.ndarray,
	global_ff: float,
	clip_range: tuple[float, float] | None = (0, 10),
) -> np.ndarray:
	log_ratio = np.clip(log_pdf_sr - log_pdf_ar, -700.0, 700.0)
	ff = global_ff * np.exp(log_ratio)
	if clip_range is None:
		return ff
	return np.clip(ff, clip_range[0], clip_range[1])


def _find_training_root_for_tag(base_root: Path, resolved_tag: str) -> Path | None:
	if not base_root.exists():
		return None

	mode_root = base_root / mode_dir
	if not mode_root.exists():
		return None

	direct = mode_root / resolved_tag
	if direct.exists():
		return direct

	prefixed = mode_root / f'training_{resolved_tag}'
	if prefixed.exists():
		return prefixed

	candidates = sorted(mode_root.glob(f'{resolved_tag}*'), key=lambda p: p.stat().st_mtime, reverse=True)
	if candidates:
		return candidates[0]

	candidates = sorted(mode_root.glob(f'training_{resolved_tag}*'), key=lambda p: p.stat().st_mtime, reverse=True)
	if candidates:
		return candidates[0]

	return None


def _first_existing(paths: list[Path]) -> Path | None:
	for p in paths:
		if p.exists():
			return p
	return None


def load_models(
	training_dir: Path,
	dim: int,
	device: t.device,
	require_full_set: bool = True,
) -> dict[str, ConditionalRealNVP]:
	base = training_dir / 'Wjets'

	if require_full_set:
		checkpoint_dirs = {
			'AR_like': _first_existing([
				base / 'DR' / 'AR-like' / 'latest',
				base / 'all' / 'AR-like' / 'latest',
				base / 'AR-like' / 'latest',
			]),
			'SR_like': _first_existing([
				base / 'DR' / 'SR-like' / 'latest',
				base / 'all' / 'SR-like' / 'latest',
				base / 'SR-like' / 'latest',
			]),
			'AR': _first_existing([
				base / 'antiDR' / 'AR' / 'latest',
				base / 'AR' / 'latest',
			]),
			'SR': _first_existing([
				base / 'antiDR' / 'SR' / 'latest',
				base / 'SR' / 'latest',
			]),
		}
	else:
		# Reduced training can be DR-only: AR-like/SR-like models without antiDR AR/SR.
		checkpoint_dirs = {
			'AR_like': _first_existing([
				base / 'DR' / 'AR-like' / 'latest',
				base / 'all' / 'AR-like' / 'latest',
				base / 'AR-like' / 'latest',
			]),
			'SR_like': _first_existing([
				base / 'DR' / 'SR-like' / 'latest',
				base / 'all' / 'SR-like' / 'latest',
				base / 'SR-like' / 'latest',
			]),
		}

	models: dict[str, ConditionalRealNVP] = {}
	for key, chk in checkpoint_dirs.items():
		if chk is None:
			raise FileNotFoundError(f'Missing checkpoint directory for {key} in {training_dir}')
		cfg = load_saved_model_config(chk, config_path)
		models[key] = load_conditional_flow(
			dim=dim,
			cfg=cfg,
			checkpoint_path=chk / 'model_checkpoint.pth',
			device=device,
		)

	return models


def plot_ff_dr_comparison(
	ff_dr_mc: np.ndarray,
	ff_dr_reduced: np.ndarray,
	plot_dir: Path,
	bins: int = 60,
	plot_range: tuple[float, float] = (1e-3, 20),
) -> None:
	edges = np.logspace(np.log10(plot_range[0]), np.log10(plot_range[1]), bins + 1)
	plt.figure(figsize=(8, 6))
	plt.hist(ff_dr_mc, bins=edges, histtype='step', color='tab:blue', label='FF_DR (MC-trained models)')
	plt.hist(ff_dr_reduced, bins=edges, histtype='step', color='tab:orange', label='FF_DR (reduced-dataset models)')
	plt.xscale('log')
	plt.xlabel('Fake Factor')
	plt.ylabel('Events')
	plt.title('FF_DR comparison on the same DR events')
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(plot_dir / 'ff_dr_mc_vs_reduced.png')
	plt.close()


def main() -> None:
	dim = len(variables)
	resolved_tag = resolve_training_name(variables)
	device = t.device('cuda:1' if t.cuda.is_available() else 'cpu')

	mc_root = SCRIPT_DIR / 'DR_SR_correction' / 'Training_results_MC'
	mc_training_dir = _find_training_root_for_tag(mc_root, resolved_tag)
	if mc_training_dir is None:
		raise FileNotFoundError(f'Could not find MC training directory for tag {resolved_tag} in {mc_root}')

	reduced_candidates = [
		SCRIPT_DIR / 'DR_SR_correction' / 'Training_results_MC_even',
		SCRIPT_DIR / 'DR_SR_correction' / 'Training_results_MC_odd',
		SCRIPT_DIR / 'Training_results_new',
		SCRIPT_DIR / 'DR_SR_correction' / 'Training_results_new',
	]
	reduced_training_dir = None
	for candidate in reduced_candidates:
		reduced_training_dir = _find_training_root_for_tag(candidate, resolved_tag)
		if reduced_training_dir is not None:
			logger.info('Using reduced training directory: %s', reduced_training_dir)
			break

	if reduced_training_dir is None:
		raise FileNotFoundError('No reduced-dataset training directory found in known candidate locations.')

	models_mc = load_models(mc_training_dir, dim=dim, device=device, require_full_set=True)
	models_reduced = None
	reduced_load_errors: list[str] = []

	for candidate in reduced_candidates:
		candidate_training_dir = _find_training_root_for_tag(candidate, resolved_tag)
		if candidate_training_dir is None:
			continue
		try:
			candidate_models = load_models(
				candidate_training_dir,
				dim=dim,
				device=device,
				require_full_set=False,
			)
		except FileNotFoundError as exc:
			reduced_load_errors.append(f'{candidate_training_dir}: {exc}')
			logger.warning('Skipping incomplete reduced training dir %s: %s', candidate_training_dir, exc)
			continue

		reduced_training_dir = candidate_training_dir
		models_reduced = candidate_models
		logger.info('Using reduced model set from %s', reduced_training_dir)
		break

	if models_reduced is None:
		error_details = '\n'.join(reduced_load_errors) if reduced_load_errors else 'no matching training root found'
		raise FileNotFoundError(
			'Could not load a complete reduced-dataset model set. Checked candidates:\n'
			f'{error_details}'
		)

	data_complete = pd.read_feather(DATA_DIR / 'data_complete.feather')
	data_events = data_complete[data_complete['process'] == 0].copy()

	masks_config = load_masks_config(MASKS_CONFIG_PATH)
	data_presel_events = _apply_config_mask(data_events, 'mask_preselection_loose', masks_config)
	data_ar_events = _apply_config_mask(data_presel_events, 'AR', masks_config)

	data_presel_mc = _apply_config_mask(data_complete[data_complete['process'] == 1], 'mask_preselection_loose', masks_config)
	data_sr_mc = _apply_config_mask(data_presel_mc, 'SR', masks_config)
	data_ar_mc = _apply_config_mask(data_presel_mc, 'AR', masks_config)
	data_sr_like_mc = _apply_config_mask(data_presel_mc, 'SR_like', masks_config)
	data_ar_like_mc = _apply_config_mask(data_presel_mc, 'AR_like', masks_config)

	input_variables = ['njets'] + list(variables)
	x_ar_events = t.tensor(get_my_data_events(data_ar_events, input_variables)).to(device)

	global_ff_dr_mc = float(np.sum(data_sr_like_mc['weight']) / np.sum(data_ar_like_mc['weight']))
	global_ff_sr_mc = float(np.sum(data_sr_mc['weight']) / np.sum(data_ar_mc['weight']))

	log_pdf_ar_like_mc, _ = evaluate_log_pdf(models_mc['AR_like'], x_ar_events)
	log_pdf_sr_like_mc, _ = evaluate_log_pdf(models_mc['SR_like'], x_ar_events)
	ff_dr_mc_raw = compute_fake_factors(log_pdf_ar_like_mc, log_pdf_sr_like_mc, global_ff_dr_mc, clip_range=None)

	log_pdf_ar_mc_on_ar, _ = evaluate_log_pdf(models_mc['AR'], x_ar_events)
	log_pdf_sr_mc_on_ar, _ = evaluate_log_pdf(models_mc['SR'], x_ar_events)
	ff_sr_mc_on_ar_raw = compute_fake_factors(log_pdf_ar_mc_on_ar, log_pdf_sr_mc_on_ar, global_ff_sr_mc, clip_range=None)

	log_pdf_ar_like_reduced, _ = evaluate_log_pdf(models_reduced['AR_like'], x_ar_events)
	log_pdf_sr_like_reduced, _ = evaluate_log_pdf(models_reduced['SR_like'], x_ar_events)
	ff_dr_reduced_raw = compute_fake_factors(log_pdf_ar_like_reduced, log_pdf_sr_like_reduced, global_ff_dr_mc, clip_range=None)

	plot_ff_dr_comparison(ff_dr_mc_raw, ff_dr_reduced_raw, plot_dir=PLOT_ROOT)

	logger.info('Saved all comparison plots to %s', PLOT_ROOT)


if __name__ == '__main__':
	main()

