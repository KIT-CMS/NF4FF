from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

CONFIG_DIR = PROJECT_ROOT / 'configs'
DATA_DIR = WORKSPACE_ROOT / 'data'
MASKS_CONFIG_PATH = CONFIG_DIR / 'masks_MC.yaml'
CONFIG_NF_PATH = CONFIG_DIR / 'config_NF.yaml'

if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from classes.Collection import ModelConfig, load_conditional_flow, load_model_config
from classes.Dataclasses import _component_collection
from classes.NeuralNetworks import ConditionalRealNVP
from CustomLogging import setup_logging


logger = setup_logging(logger=logging.getLogger(__name__))

MODE_DIR = 'conditional_njets_input'
TRAINING_ROOT = SCRIPT_DIR / 'Training_results_MC'
FF_FLOW_ROOT = SCRIPT_DIR / 'FF_conditioned_models'
OUTPUT_DIR = SCRIPT_DIR / 'FF_comparison_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_my_data(df: pd.DataFrame, training_var: list[str]) -> _component_collection:
	return _component_collection(
		X=df[training_var].to_numpy(dtype=np.float32),
		weights=df['weight'].to_numpy(dtype=np.float32),
	)

def get_my_data_wjets(df: pd.DataFrame, training_var: list[str]) -> _component_collection:
	_df = df
	return _component_collection(
		X=_df[training_var].to_numpy(dtype=np.float32),
		weights=_df['weight_wjets'].to_numpy(dtype=np.float32),
	)


def resolve_training_name(variables: list[str]) -> str:
	tail = variables[4:]
	tag = '_'.join(tail) if tail else 'none'
	return f'training_vars{len(variables)}_{tag}'


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


def compute_fake_factors(log_pdf_ar: np.ndarray, log_pdf_sr: np.ndarray, global_ff: float) -> np.ndarray:
	log_ratio = np.clip(log_pdf_sr - log_pdf_ar, -700.0, 700.0)
	return np.clip(global_ff * np.exp(log_ratio), 0, 10)


@t.no_grad()
def evaluate_log_pdf(model: ConditionalRealNVP, X: t.Tensor) -> tuple[np.ndarray, np.ndarray]:
	cond_dim = int(getattr(model, 'cond_dim', 0))
	x_features = X[:, cond_dim:]
	x_preprocessed, _, valid_mask = model.apply_preprocessing(x_features)
	Xs = model.apply_scaler(x_preprocessed)

	n_nan_after_scale = t.isnan(Xs[valid_mask]).any(dim=-1).sum().item() if valid_mask.any() else 0
	n_inf_after_scale = t.isinf(Xs[valid_mask]).any(dim=-1).sum().item() if valid_mask.any() else 0
	if n_nan_after_scale > 0 or n_inf_after_scale > 0:
		logger.warning('evaluate_log_pdf: nan_after_scale=%d inf_after_scale=%d', n_nan_after_scale, n_inf_after_scale)

	log_pdf = model(X)
	log_pdf = t.nan_to_num(log_pdf, nan=-700.0, neginf=-700.0, posinf=700.0)
	return log_pdf.detach().cpu().numpy(), valid_mask.detach().cpu().numpy()


def load_ff_conditional_flow(checkpoint_dir: Path, device: t.device) -> ConditionalRealNVP:
	config_path = checkpoint_dir.parent / 'config.yaml'
	if not config_path.exists():
		raise FileNotFoundError(f'FF flow config not found: {config_path}')

	with open(config_path, 'r') as handle:
		cfg = yaml.safe_load(handle) or {}

	model = ConditionalRealNVP(
		dim=int(cfg.get('dim', 1)),
		cond_dim=int(cfg.get('cond_dim', 1)),
		n_layers=int(cfg.get('n_layers', 6)),
		hidden_dims=(int(cfg.get('hidden_dims', 128)),),
		s_scale=float(cfg.get('s_scale', 2.0)),
		use_cut_preprocessing=bool(cfg.get('use_cut_preprocessing', False)),
		use_tail_preprocessing=bool(cfg.get('use_tail_preprocessing', False)),
	).to(device)

	ckpt = t.load(checkpoint_dir / 'model_checkpoint.pth', map_location=device)
	model.load_state_dict(ckpt['model_state_dict'])
	model.eval()
	return model


@t.no_grad()
def sample_ff_sr_from_flow(model: ConditionalRealNVP, ff_dr: np.ndarray, device: t.device) -> np.ndarray:
	cond = t.tensor(ff_dr, dtype=t.float32, device=device).reshape(-1, 1)
	samples = model.sample(cond).reshape(-1)
	return np.clip(samples.detach().cpu().numpy(), 0.0, 10.0)


def plot_ff_distributions(
	ff_sr: np.ndarray,
	ff_dr: np.ndarray,
	ff_sr_flow: np.ndarray,
	plot_dir: Path,
	bins: int = 60,
	range: tuple[float, float] = (0.001, 10),
) -> None:
	plot_dir.mkdir(parents=True, exist_ok=True)
	edges = np.logspace(np.log10(range[0]), np.log10(range[1]), bins + 1)
	plt.figure(figsize=(8, 6))
	plt.hist(ff_sr, bins=edges, histtype='step', lw=1.7, color='tab:blue', label='FF_SR (from base NF)')
	plt.hist(ff_dr, bins=edges, histtype='step', lw=1.7, color='tab:red', label='FF_DR')
	plt.hist(ff_sr_flow, bins=edges, histtype='step', lw=1.7, ls='--', color='tab:green', label='FF_SR_flow (cond. flow)')
	plt.xscale('log')
	plt.xlabel('Fake Factor')
	plt.ylabel('Frequency')
	plt.title('Fake Factor comparison')
	plt.grid(True, linestyle='--', alpha=0.4)
	plt.legend()
	plt.tight_layout()
	plt.savefig(plot_dir / 'fake_factor_distribution_comparison.png')
	plt.savefig(plot_dir / 'fake_factor_distribution_comparison.pdf')
	plt.close()


def main() -> None:
	t.manual_seed(42)
	np.random.seed(42)

	with open(CONFIG_DIR / 'training_variables.yaml', 'r') as f:
		variables = yaml.safe_load(f)['variables_MC']

	dim = len(variables)
	resolved_tag = resolve_training_name(variables)
	training_root = TRAINING_ROOT / MODE_DIR / resolved_tag
	masks_config = load_masks_config(MASKS_CONFIG_PATH)
	device = t.device('cuda:1' if t.cuda.is_available() else 'cpu')

	chk_pth_model_AR_like = training_root / 'Wjets' / 'DR' / 'AR-like' / 'latest'
	chk_pth_model_SR_like = training_root / 'Wjets' / 'DR' / 'SR-like' / 'latest'
	chk_pth_model_AR = training_root / 'Wjets' / 'antiDR' / 'AR' / 'latest'
	chk_pth_model_SR = training_root / 'Wjets' / 'antiDR' / 'SR' / 'latest'

	config_AR_like = load_saved_model_config(chk_pth_model_AR_like, CONFIG_NF_PATH)
	config_SR_like = load_saved_model_config(chk_pth_model_SR_like, CONFIG_NF_PATH)
	config_AR = load_saved_model_config(chk_pth_model_AR, CONFIG_NF_PATH)
	config_SR = load_saved_model_config(chk_pth_model_SR, CONFIG_NF_PATH)

	model_AR_like = load_conditional_flow(dim=dim, cfg=config_AR_like, checkpoint_path=chk_pth_model_AR_like / 'model_checkpoint.pth', device=device)
	model_SR_like = load_conditional_flow(dim=dim, cfg=config_SR_like, checkpoint_path=chk_pth_model_SR_like / 'model_checkpoint.pth', device=device)
	model_AR = load_conditional_flow(dim=dim, cfg=config_AR, checkpoint_path=chk_pth_model_AR / 'model_checkpoint.pth', device=device)
	model_SR = load_conditional_flow(dim=dim, cfg=config_SR, checkpoint_path=chk_pth_model_SR / 'model_checkpoint.pth', device=device)

	data_complete = pd.read_feather(DATA_DIR / 'data_complete.feather')
	data_MC = data_complete[data_complete['process'] == 1]
	data_events = data_complete[data_complete['process'] == 0]

	data_presel_events = _apply_config_mask(data_events, 'mask_preselection_loose', masks_config)
	data_sr = _apply_config_mask(data_presel_events, 'SR', masks_config)
	data_ar = _apply_config_mask(data_presel_events, 'AR', masks_config)	

	data_presel_MC = _apply_config_mask(data_MC, 'mask_preselection_loose', masks_config)
	data_sr_MC = _apply_config_mask(data_presel_MC, 'SR', masks_config)
	data_ar_MC = _apply_config_mask(data_presel_MC, 'AR', masks_config)
	data_sr_like_MC = _apply_config_mask(data_presel_MC, 'SR_like', masks_config)
	data_ar_like_MC = _apply_config_mask(data_presel_MC, 'AR_like', masks_config)

	input_variables = ['njets'] + list(variables)

	X_AR = get_my_data_wjets(data_ar, input_variables).to_torch(device=None).X.to(device)

	# Same cross-region setup as in FF_calculation.py
	log_pdf_ar_like, _ = evaluate_log_pdf(model_AR_like, X_AR)
	log_pdf_sr_like, _ = evaluate_log_pdf(model_SR_like, X_AR)
	log_pdf_ar, _ = evaluate_log_pdf(model_AR, X_AR)
	log_pdf_sr, _ = evaluate_log_pdf(model_SR, X_AR)

	global_ff_dr = np.sum(data_sr_like_MC['weight']) / np.sum(data_ar_like_MC['weight'])
	global_ff_sr = np.sum(data_sr_MC['weight']) / np.sum(data_ar_MC['weight'])

	ff_dr = compute_fake_factors(log_pdf_ar_like, log_pdf_sr_like, global_ff_dr)
	ff_sr = compute_fake_factors(log_pdf_ar, log_pdf_sr, global_ff_sr)

	# Load conditional FF flow and sample FF_SR_flow conditioned on FF_DR
	ff_flow_ckpt = FF_FLOW_ROOT / 'MC_ff_sr_given_ff_dr' / 'latest'
	ff_flow_model = load_ff_conditional_flow(ff_flow_ckpt, device=device)
	ff_sr_flow = sample_ff_sr_from_flow(ff_flow_model, ff_dr, device=device)

	logger.info('FF means: DR=%.4f SR=%.4f SR_flow=%.4f', float(np.mean(ff_dr)), float(np.mean(ff_sr)), float(np.mean(ff_sr_flow)))
	logger.info('FF medians: DR=%.4f SR=%.4f SR_flow=%.4f', float(np.median(ff_dr)), float(np.median(ff_sr)), float(np.median(ff_sr_flow)))

	plot_ff_distributions(ff_sr, ff_dr, ff_sr_flow, OUTPUT_DIR)
	logger.info('Saved comparison plot to %s', OUTPUT_DIR)


if __name__ == '__main__':
	main()

