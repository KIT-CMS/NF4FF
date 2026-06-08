from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

CONFIG_DIR = PROJECT_ROOT / 'configs'
DATA_DIR = WORKSPACE_ROOT / 'data'
TRAINING_RESULTS_ROOT = SCRIPT_DIR / 'Training_results_MC'

if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from classes.Collection import (
	ModelConfig,
	load_conditional_flow,
	load_model_config,
	evaluate_pdf,
	compute_eventwise_fake_factors,
	get_my_data_wjets,
	get_my_data_qcd,
)
from classes.Dataclasses import _component_collection
from classes.NeuralNetworks import ConditionalRealNVP
from CustomLogging import setup_logging, LogContext


logger = setup_logging(logger=logging.getLogger(__name__))
log = LogContext(logger)

CONFIG_NF_PATH = CONFIG_DIR / 'config_DR_SR_correction_NF.yaml'
TRAINING_VARIABLES_PATH = CONFIG_DIR / 'training_variables.yaml'
MASKS_CONFIG_PATH = CONFIG_DIR / 'masks_MC.yaml'
FF_COND_OUTPUT_ROOT = SCRIPT_DIR / 'FF_conditioned_models'


def _evaluate_ff_loader(model: ConditionalRealNVP, loader: DataLoader, device: t.device) -> float:
	model.eval()
	loss_sum = 0.0
	n_events = 0
	use_amp = (device.type == 'cuda')

	with t.no_grad():
		for xb in loader:
			x = xb[0].to(device, non_blocking=True)
			with t.amp.autocast('cuda', enabled=use_amp):
				log_px = model(x).reshape(-1)
				loss = -(log_px).sum()
			loss_sum += loss.item()
			n_events += x.shape[0]

	return loss_sum / max(n_events, 1)


def _build_ff_conditional_model(base_cfg: ModelConfig, device: t.device) -> ConditionalRealNVP:
	# Learn p(ff_SR | ff_DR): condition=ff_DR (1 dim), modeled feature=ff_SR (1 dim).
	model = ConditionalRealNVP(
		dim=1,
		cond_dim=1,
		n_layers=base_cfg.n_layers,
		hidden_dims=(base_cfg.hidden_dims,),
		s_scale=base_cfg.s_scale,
		# FF values are small (typically O(1)); turn off cut preprocessing thresholds.
		use_cut_preprocessing=False,
		use_tail_preprocessing=False,
	).to(device)
	return model


def train_conditional_ff_flow(
	ff_dr: np.ndarray,
	ff_sr: np.ndarray,
	label: str,
	device: t.device,
	config_path: Path = CONFIG_NF_PATH,
	output_root: Path = FF_COND_OUTPUT_ROOT,
	epochs: int = 200,
	test_size: float = 0.25,
	batch_size: int = 2048,
	patience: int = 30,
	learning_rate: float = 1e-3,
	weight_decay: float = 1e-5,
) -> Path:
	ff_dr = np.asarray(ff_dr, dtype=np.float32)
	ff_sr = np.asarray(ff_sr, dtype=np.float32)

	if ff_dr.shape[0] != ff_sr.shape[0]:
		n = min(ff_dr.shape[0], ff_sr.shape[0])
		logger.warning(
			'[%s] ff_dr and ff_sr length mismatch (%d vs %d). Downsampling to %d paired entries.',
			label,
			ff_dr.shape[0],
			ff_sr.shape[0],
			n,
		)
		rng = np.random.default_rng(42)
		ff_dr = ff_dr[rng.choice(ff_dr.shape[0], size=n, replace=False)]
		ff_sr = ff_sr[rng.choice(ff_sr.shape[0], size=n, replace=False)]

	valid = np.isfinite(ff_dr) & np.isfinite(ff_sr)
	ff_dr = ff_dr[valid]
	ff_sr = ff_sr[valid]
	if ff_dr.size < 100:
		raise ValueError(f'Not enough valid FF pairs for training ({ff_dr.size}).')

	X = np.stack([ff_dr, ff_sr], axis=1).astype(np.float32)

	rng = np.random.default_rng(42)
	idx = rng.permutation(X.shape[0])
	n_val = max(1, int(test_size * X.shape[0]))
	val_idx, train_idx = idx[:n_val], idx[n_val:]
	X_train = t.tensor(X[train_idx], dtype=t.float32)
	X_val = t.tensor(X[val_idx], dtype=t.float32)

	train_loader = DataLoader(
		TensorDataset(X_train),
		batch_size=min(batch_size, max(1, X_train.shape[0])),
		shuffle=True,
		pin_memory=True,
		num_workers=0,
	)
	val_loader = DataLoader(
		TensorDataset(X_val),
		batch_size=min(batch_size, max(1, X_val.shape[0])),
		shuffle=False,
		pin_memory=True,
		num_workers=0,
	)

	base_cfg = load_model_config(str(config_path))
	model = _build_ff_conditional_model(base_cfg, device=device)

	with t.no_grad():
		x_features_train = X_train[:, 1:]
		shift = x_features_train.mean(dim=0)
		scale = x_features_train.std(dim=0, unbiased=False).clamp_min(1e-12)
	model.initialize_scaler(shift, scale)

	optimizer = t.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
	use_amp = (device.type == 'cuda')
	scaler = t.amp.GradScaler('cuda', enabled=use_amp)

	best_val_nll = float('inf')
	best_state = None
	stale_epochs = 0
	history_rows: list[dict] = []

	with log.training_dashboard() as dash:
		for epoch in range(1, epochs + 1):
			model.train()
			train_loss_sum = 0.0
			n_train = 0

			for xb in train_loader:
				x = xb[0].to(device, non_blocking=True)
				optimizer.zero_grad(set_to_none=True)
				with t.amp.autocast('cuda', enabled=use_amp):
					log_px = model(x).reshape(-1)
					loss = -(log_px).mean()

				scaler.scale(loss).backward()
				nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
				scaler.step(optimizer)
				scaler.update()

				train_loss_sum += float(loss.item()) * x.shape[0]
				n_train += x.shape[0]

			avg_train_nll = train_loss_sum / max(n_train, 1)
			avg_val_nll = _evaluate_ff_loader(model, val_loader, device)
			scheduler.step(avg_val_nll)

			history_rows.append(
				{
					'epoch': epoch,
					'train_loss': avg_train_nll,
					'val_loss': avg_val_nll,
					'lr': scheduler.get_last_lr()[0],
				}
			)

			if avg_val_nll < best_val_nll:
				best_val_nll = avg_val_nll
				stale_epochs = 0
				best_state = {
					'model_state_dict': model.state_dict(),
					'variables': ['ff_DR', 'ff_SR'],
					'schema': 'conditional_ff_sr_given_ff_dr_v1',
					'training_label': label,
					'best_val_nll': best_val_nll,
				}
			else:
				stale_epochs += 1

			dash.update(
				epoch=epoch,
				train_loss=np.round(avg_train_nll, 6),
				val_loss=np.round(avg_val_nll, 6),
				lr=scheduler.get_last_lr()[0],
				region=f'FF_conditional {label}',
			)

			if stale_epochs >= patience:
				logger.info('[%s] Early stopping at epoch %d (best val_nll=%.6f)', label, epoch, best_val_nll)
				break

	if best_state is None:
		raise RuntimeError(f'No checkpoint was created for FF conditional flow training [{label}].')

	out_dir = output_root / label
	latest_dir = out_dir / 'latest'
	latest_dir.mkdir(parents=True, exist_ok=True)

	t.save(best_state, out_dir / 'model_checkpoint.pth')
	t.save(best_state, latest_dir / 'model_checkpoint.pth')
	pd.DataFrame(history_rows).to_pickle(out_dir / 'training_logs.pkl')
	pd.DataFrame(history_rows).to_pickle(latest_dir / 'training_logs.pkl')

	with open(out_dir / 'config.yaml', 'w') as handle:
		yaml.safe_dump(
			{
				'n_layers': int(base_cfg.n_layers),
				'hidden_dims': int(base_cfg.hidden_dims),
				's_scale': float(base_cfg.s_scale),
				'cond_dim': 1,
				'dim': 1,
				'use_cut_preprocessing': False,
				'use_tail_preprocessing': False,
			},
			handle,
		)

	logger.info('[%s] Saved FF conditional flow to %s (best val_nll=%.6f)', label, out_dir, best_val_nll)
	return out_dir


def get_my_data(df: pd.DataFrame, training_var: list[str]) -> _component_collection:
	return _component_collection(
		X=df[training_var].to_numpy(dtype=np.float32),
		weights=df['weight'].to_numpy(dtype=np.float32),
	)


def load_training_variables(path: Path = TRAINING_VARIABLES_PATH) -> list[str]:
	with open(path, 'r') as handle:
		raw = yaml.safe_load(handle) or {}

	variables = raw.get('training_variables_MC', raw.get('variables_MC'))
	if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
		raise ValueError(
			f"Expected key 'training_variables_MC' (or 'variables_MC') with list[str] in {path}."
		)
	return variables


def resolve_training_name(variables: list[str]) -> str:
	tail = variables[4:]
	tag = '_'.join(tail) if tail else 'none'
	return f'training_vars{len(variables)}_{tag}'


def resolve_mode_dir(base_path: Path) -> str:
	# Keep compatibility with naming variants used in this project.
	candidates = ['conditional_input', 'conditional_njets_input']
	for candidate in candidates:
		if (base_path / candidate).exists():
			return candidate
	raise FileNotFoundError(
		f"Could not find any of {candidates} in {base_path}."
	)


def load_saved_model_config(checkpoint_dir: Path, fallback_path: Path) -> ModelConfig:
	saved_cfg = checkpoint_dir.parent / 'config.yaml'
	if saved_cfg.exists():
		return load_model_config(str(saved_cfg))

	logger.warning('No local config.yaml for %s, using fallback config %s', checkpoint_dir, fallback_path)
	return load_model_config(str(fallback_path))


def load_masks_config(path: Path = MASKS_CONFIG_PATH) -> dict[str, list[str]]:
	if not path.exists():
		raise FileNotFoundError(f'Mask configuration not found: {path}')

	with open(path, 'r') as handle:
		raw = yaml.safe_load(handle) or {}

	masks = raw.get('masks', raw)
	if not isinstance(masks, dict):
		raise ValueError(f'Invalid masks configuration in {path}: expected mapping at top level.')

	normalized: dict[str, list[str]] = {}
	for name, expressions in masks.items():
		if isinstance(expressions, str):
			normalized[name] = [expressions]
		elif isinstance(expressions, list) and all(isinstance(expr, str) for expr in expressions):
			normalized[name] = expressions
		else:
			raise TypeError(
				f"Mask '{name}' in {path} must be str or list[str], got {type(expressions).__name__}."
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
	x_scaled = model.apply_scaler(x_preprocessed)

	if valid_mask.any():
		n_nan_after_scale = t.isnan(x_scaled[valid_mask]).any(dim=-1).sum().item()
		n_inf_after_scale = t.isinf(x_scaled[valid_mask]).any(dim=-1).sum().item()
		if n_nan_after_scale > 0 or n_inf_after_scale > 0:
			logger.warning(
				'Scaled inputs contain invalid values: nan=%d inf=%d',
				n_nan_after_scale,
				n_inf_after_scale,
			)

	log_pdf = model(X)
	log_pdf = t.nan_to_num(log_pdf, nan=-700.0, neginf=-700.0, posinf=700.0)

	return log_pdf.detach().cpu().numpy(), valid_mask.detach().cpu().numpy()


def compute_fake_factors(log_pdf_ar: np.ndarray, log_pdf_sr: np.ndarray, global_ff: float) -> np.ndarray:
	# Same FF logic as FF_calculation.py:
	# FF = global_ff * exp(log p_SR - log p_AR)
	log_ratio = np.clip(log_pdf_sr - log_pdf_ar, -700.0, 700.0)
	return np.clip(global_ff * np.exp(log_ratio), 0.0, 10.0)


def load_four_models_from_training_variables(
	variables: list[str],
	training_results_root: Path = TRAINING_RESULTS_ROOT,
	config_path: Path = CONFIG_NF_PATH,
	device: t.device | None = None,
	model_prefix: str = '',
) -> dict[str, ConditionalRealNVP]:
	"""
	Load four conditional models (AR_like, SR_like, AR, SR) from Training_results_MC.
	
	Args:
		variables: List of training variables
		training_results_root: Root directory containing training results
		config_path: Path to NF config
		device: torch device
		model_prefix: Optional prefix for logging (e.g., 'MC_' or 'reduced_')
	
	Returns:
		Dictionary with keys: 'AR_like', 'SR_like', 'AR', 'SR'
	"""
	if device is None:
		device = t.device('cuda' if t.cuda.is_available() else 'cpu')

	mode_dir = resolve_mode_dir(training_results_root)
	training_tag = resolve_training_name(variables)
	training_root = training_results_root / mode_dir / training_tag / 'Wjets'

	checkpoint_dirs = {
		'AR_like': training_root / 'DR' / 'AR-like' / 'latest',
		'SR_like': training_root / 'DR' / 'SR-like' / 'latest',
		'AR': training_root / 'antiDR' / 'AR' / 'latest',
		'SR': training_root / 'antiDR' / 'SR' / 'latest',
	}

	dim = len(variables)
	models: dict[str, ConditionalRealNVP] = {}
	for key, checkpoint_dir in checkpoint_dirs.items():
		checkpoint_path = checkpoint_dir / 'model_checkpoint.pth'
		if not checkpoint_path.exists():
			raise FileNotFoundError(f'Model checkpoint does not exist: {checkpoint_path}')

		model_cfg = load_saved_model_config(checkpoint_dir, config_path)
		models[key] = load_conditional_flow(
			dim=dim,
			cfg=model_cfg,
			checkpoint_path=str(checkpoint_path),
			device=device,
			cond_dim=1,
		)

	logger.info('Loaded %sfour models from %s', f'{model_prefix} ' if model_prefix else '', training_root)
	return models


def main() -> None:
	variables = load_training_variables(TRAINING_VARIABLES_PATH)
	device = t.device('cuda' if t.cuda.is_available() else 'cpu')

	# 1) Load the four MC models according to training_variables_MC.
	models_mc = load_four_models_from_training_variables(
		variables=variables,
		device=device,
		model_prefix='MC',
	)

	# 2) Try loading reduced/alternative models from known candidate roots.
	models_reduced = None
	reduced_roots = [
		SCRIPT_DIR / 'Training_results_MC_even',
		SCRIPT_DIR / 'Training_results_MC_odd',
		PROJECT_ROOT / 'src' / 'Training_results_new',
	]
	for reduced_root in reduced_roots:
		if not reduced_root.exists():
			continue
		try:
			models_reduced = load_four_models_from_training_variables(
				variables=variables,
				training_results_root=reduced_root,
				config_path=CONFIG_NF_PATH,
				device=device,
				model_prefix=f'reduced_dataset({reduced_root.name})',
			)
			logger.info('Using reduced dataset models from %s', reduced_root)
			break
		except Exception as e:
			logger.info('Skipping reduced root %s: %s', reduced_root, e)

	if models_reduced is None:
		logger.info('No compatible reduced-dataset model set found. Proceeding with MC models only.')

	# 3) Load masks config and data
	masks_config = load_masks_config(MASKS_CONFIG_PATH)
	data_complete = pd.read_feather(DATA_DIR / 'data_complete.feather')
	data_events = data_complete[data_complete['process'] == 0]
	data_MC = data_complete[data_complete['process'] == 1]
	# 4) Apply masks and prepare data for both MC and real (reduced) data
	logger.info('Applying masks to data...')

	data_presel = _apply_config_mask(data_complete, 'mask_preselection_loose', masks_config)
	data_sr = _apply_config_mask(data_presel, 'SR', masks_config)
	data_ar = _apply_config_mask(data_presel, 'AR', masks_config)
	data_sr_like = _apply_config_mask(data_presel, 'SR_like', masks_config)
	data_ar_like = _apply_config_mask(data_presel, 'AR_like', masks_config)

	input_variables = ['njets'] + list(variables)
	
	# 5) Calculate FF using MC models
	logger.info('Computing fake factors with MC-trained models...')
	x_ar = get_my_data(data_ar, input_variables).to_torch(device=None).X.to(device)
	x_ar_like = get_my_data(data_ar_like, input_variables).to_torch(device=None).X.to(device)

	# FF_DR: DR-trained models applied to antiDR (SR region) events
	log_pdf_ar_like_mc, valid_ar_like_mc = evaluate_log_pdf(models_mc['AR_like'], x_ar)
	log_pdf_sr_like_mc, valid_sr_like_mc = evaluate_log_pdf(models_mc['SR_like'], x_ar)
	
	# FF_SR: antiDR-trained models applied to DR region events
	log_pdf_ar_mc, valid_ar_mc = evaluate_log_pdf(models_mc['AR'], x_ar_like)
	log_pdf_sr_mc, valid_sr_mc = evaluate_log_pdf(models_mc['SR'], x_ar_like)

	global_ff_dr_mc = float(np.sum(data_sr_like['weight']) / np.sum(data_ar_like['weight']))
	global_ff_sr_mc = float(np.sum(data_sr['weight']) / np.sum(data_ar['weight']))

	ff_dr_mc = compute_fake_factors(log_pdf_ar_like_mc, log_pdf_sr_like_mc, global_ff_dr_mc)
	ff_sr_mc = compute_fake_factors(log_pdf_ar_mc, log_pdf_sr_mc, global_ff_sr_mc)

	# Build paired target on the SAME events as ff_dr_mc (x_ar) for conditional training.
	log_pdf_ar_mc_on_ar, _ = evaluate_log_pdf(models_mc['AR'], x_ar)
	log_pdf_sr_mc_on_ar, _ = evaluate_log_pdf(models_mc['SR'], x_ar)
	ff_sr_mc_paired = compute_fake_factors(log_pdf_ar_mc_on_ar, log_pdf_sr_mc_on_ar, global_ff_sr_mc)

	logger.info('MC models: Loaded variables: %s', variables)
	logger.info('MC models: Valid preprocessing fractions: DR(AR-like=%.4f, SR-like=%.4f), antiDR(AR=%.4f, SR=%.4f)',
				valid_ar_like_mc.mean(), valid_sr_like_mc.mean(), valid_ar_mc.mean(), valid_sr_mc.mean())
	logger.info('MC models: Global FF DR=%.6f, SR=%.6f', global_ff_dr_mc, global_ff_sr_mc)
	logger.info('MC models: FF DR mean=%.6f median=%.6f', float(np.mean(ff_dr_mc)), float(np.median(ff_dr_mc)))
	logger.info('MC models: FF SR mean=%.6f median=%.6f', float(np.mean(ff_sr_mc)), float(np.median(ff_sr_mc)))

	# 5b) Train conditional flow: learn p(ff_SR | ff_DR) on MC-derived FF pairs.
	train_conditional_ff_flow(
		ff_dr=ff_dr_mc,
		ff_sr=ff_sr_mc_paired,
		label='MC_ff_sr_given_ff_dr',
		device=device,
	)

	# 6) Calculate FF using reduced dataset models (if available)
	if models_reduced is not None:
		logger.info('Computing fake factors with reduced-dataset-trained models...')
		# FF_DR: DR-trained models applied to antiDR (SR region) events
		log_pdf_ar_like_reduced, valid_ar_like_reduced = evaluate_log_pdf(models_reduced['AR_like'], x_ar)
		log_pdf_sr_like_reduced, valid_sr_like_reduced = evaluate_log_pdf(models_reduced['SR_like'], x_ar)
		
		# FF_SR: antiDR-trained models applied to DR region events
		log_pdf_ar_reduced, valid_ar_reduced = evaluate_log_pdf(models_reduced['AR'], x_ar_like)
		log_pdf_sr_reduced, valid_sr_reduced = evaluate_log_pdf(models_reduced['SR'], x_ar_like)

		ff_dr_reduced = compute_fake_factors(log_pdf_ar_like_reduced, log_pdf_sr_like_reduced, global_ff_dr_mc)
		ff_sr_reduced = compute_fake_factors(log_pdf_ar_reduced, log_pdf_sr_reduced, global_ff_sr_mc)

		# Build paired target on the SAME events as ff_dr_reduced (x_ar).
		log_pdf_ar_reduced_on_ar, _ = evaluate_log_pdf(models_reduced['AR'], x_ar)
		log_pdf_sr_reduced_on_ar, _ = evaluate_log_pdf(models_reduced['SR'], x_ar)
		ff_sr_reduced_paired = compute_fake_factors(log_pdf_ar_reduced_on_ar, log_pdf_sr_reduced_on_ar, global_ff_sr_mc)

		logger.info('Reduced dataset models: Valid preprocessing fractions: DR(AR-like=%.4f, SR-like=%.4f), antiDR(AR=%.4f, SR=%.4f)',
					valid_ar_like_reduced.mean(), valid_sr_like_reduced.mean(), valid_ar_reduced.mean(), valid_sr_reduced.mean())
		logger.info('Reduced dataset models: FF DR mean=%.6f median=%.6f', float(np.mean(ff_dr_reduced)), float(np.median(ff_dr_reduced)))
		logger.info('Reduced dataset models: FF SR mean=%.6f median=%.6f', float(np.mean(ff_sr_reduced)), float(np.median(ff_sr_reduced)))

		# 6b) Train conditional flow: learn p(ff_SR | ff_DR) on reduced-dataset FF pairs.
		train_conditional_ff_flow(
			ff_dr=ff_dr_reduced,
			ff_sr=ff_sr_reduced_paired,
			label='reduced_ff_sr_given_ff_dr',
			device=device,
		)

		# Compare MC vs reduced dataset FF
		logger.info('=== MC vs Reduced Dataset FF Comparison ===')
		logger.info('DR FF ratio (reduced/MC) - mean: %.4f, median: %.4f, std: %.4f',
					float(np.mean(ff_dr_reduced / np.maximum(ff_dr_mc, 1e-6))),
					float(np.median(ff_dr_reduced / np.maximum(ff_dr_mc, 1e-6))),
					float(np.std(ff_dr_reduced / np.maximum(ff_dr_mc, 1e-6))))
		logger.info('SR FF ratio (reduced/MC) - mean: %.4f, median: %.4f, std: %.4f',
					float(np.mean(ff_sr_reduced / np.maximum(ff_sr_mc, 1e-6))),
					float(np.median(ff_sr_reduced / np.maximum(ff_sr_mc, 1e-6))),
					float(np.std(ff_sr_reduced / np.maximum(ff_sr_mc, 1e-6))))


if __name__ == '__main__':
	main()

