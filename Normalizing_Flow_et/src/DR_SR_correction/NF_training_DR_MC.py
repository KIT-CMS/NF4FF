import logging
import random
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import yaml
from tap import Tap
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / 'configs'
DATA_DIR = WORKSPACE_ROOT / 'data'

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from classes.Dataclasses import RealNVP_config, _component_collection
from classes.NeuralNetworks import ConditionalRealNVP
from CustomLogging import LogContext, setup_logging


SEED = 42


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

with open(CONFIG_DIR / 'training_variables.yaml', 'r') as f:
    variables = yaml.safe_load(f)['variables_MC']

logger = setup_logging(logger=logging.getLogger(__name__))
log = LogContext(logger)

PATIENCE = 30

TRAINING_MODEL_CONDITIONAL = 'conditional_nf'


class Args(Tap):

    output_root_base: str = 'Training_results_MC'  # Base directory where training folders are written.
    test_size: float = 0.25  # Validation fraction for the train/validation split.
    random_state: int = SEED  # Random seed used for train/validation splitting.
    training_process: Literal['dr_like', 'antidr', 'both'] = 'both'  # Choose Wjets training region setup: DR-like (AR-like/SR-like), antiDR (AR/SR), or both.


@dataclass(frozen=True)
class ProcessTrainingSpec:
    name: str
    process_id: int
    region_sign_column: str
    weight_column: str
    output_root: str
    dr_mask_name: str
    ar_like_mask_name: str
    sr_like_mask_name: str
    ar_region_name: str
    sr_region_name: str
    data_getter: Callable

def get_my_data(df, training_var):
    _df = df
    return _component_collection(
        X=_df[training_var].to_numpy(dtype=np.float32),
        weights=_df["weight"].to_numpy(dtype=np.float32),
    )


# ----- shared helpers -----


MASKS_CONFIG_PATH = CONFIG_DIR / 'masks_MC.yaml'


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



def evaluate_loader(model, loader, device):
    model.eval()
    loss_sum = 0.0
    weight_sum = 0.0
    use_amp = (device.type == 'cuda')

    with t.no_grad():
        for Xb, Wb in loader:
            Xb = Xb.to(device, non_blocking=True)
            Wb = Wb.to(device, non_blocking=True)

            with t.amp.autocast('cuda', enabled=use_amp):
                log_px = model(Xb).reshape(-1)
                loss = (-(log_px) * Wb).sum()

            loss_sum += loss.item()
            weight_sum += Wb.sum().item()

    return loss_sum / max(weight_sum, 1e-12)


def build_conditional_nf(config, dim, shift, scale, device):
    model = ConditionalRealNVP(
        dim=dim,
        cond_dim=1,
        n_layers=config.n_layers,
        hidden_dims=(config.hidden_dims,),
        s_scale=config.s_scale,
        use_cut_preprocessing=config.use_cut_preprocessing,
        cut_preprocessing_index=config.cut_preprocessing_index,
        cut_preprocessing_thresholds=config.cut_preprocessing_thresholds,
        cut_preprocessing_epsilon=config.cut_preprocessing_epsilon,
        use_tail_preprocessing=config.use_tail_preprocessing,
        tail_preprocessing_index=config.tail_preprocessing_index,
        tail_preprocessing_type=config.tail_preprocessing_type,
        tail_preprocessing_center=config.tail_preprocessing_center,
        tail_preprocessing_scale=config.tail_preprocessing_scale,
        tail_preprocessing_epsilon=config.tail_preprocessing_epsilon,
    ).to(device)
    if shift is not None and scale is not None:
        model.initialize_scaler(shift, scale)
    return model


def _compute_preprocessed_scaler_stats(model, x_train: t.Tensor, uses_njets_context: bool):
    if uses_njets_context:
        x_features = x_train[:, 1:]
    else:
        x_features = x_train

    with t.no_grad():
        x_preprocessed, _, valid_mask = model.apply_preprocessing(x_features)

    if valid_mask.any():
        x_for_stats = x_preprocessed[valid_mask]
    else:
        raise RuntimeError("No valid events after preprocessing; cannot initialize scaler.")

    shift = x_for_stats.mean(dim=0)
    scale = x_for_stats.std(dim=0, unbiased=False).clamp_min(1e-12)
    valid_fraction = valid_mask.float().mean().item()
    return shift, scale, valid_fraction


def _initialize_model_scaler(model, shift: t.Tensor, scale: t.Tensor):
    model.initialize_scaler(shift, scale)


def _weight_sum(df: pd.DataFrame, weight_column: str) -> float:
    if weight_column not in df.columns:
        raise KeyError(f"Missing weight column '{weight_column}'. Available columns do not include it.")
    return float(df[weight_column].fillna(0.0).sum())


def _count_and_weight(df: pd.DataFrame, weight_column: str) -> tuple[int, float]:
    return len(df), _weight_sum(df, weight_column)


def prepare_region_samples(
    data_complete,
    spec: ProcessTrainingSpec,
    test_size: float,
    random_state: int,
    masks_config: dict[str, list[str]],
):
    if spec.weight_column not in data_complete.columns:
        raise KeyError(f"{spec.name}: weight column '{spec.weight_column}' not found in input dataframe.")

    data_dr = _apply_config_mask(data_complete, spec.dr_mask_name, masks_config)
    data_dr = data_dr[(data_dr.process == spec.process_id) & (data_dr[spec.region_sign_column] == True)].reset_index(drop=True)

    dr_count, dr_weight_sum = _count_and_weight(data_dr, spec.weight_column)
    logger.info(
        "%s cutflow after %s + process/sign filter: events=%d, weight_sum=%.6f",
        spec.name,
        spec.dr_mask_name,
        dr_count,
        dr_weight_sum,
    )
    if dr_count == 0:
        raise ValueError(
            f"{spec.name}: no events survive {spec.dr_mask_name} with process=={spec.process_id} and {spec.region_sign_column}==True."
        )
    if not np.isfinite(dr_weight_sum) or dr_weight_sum == 0.0:
        raise ValueError(
            f"{spec.name}: selected DR sample has invalid total weight for column '{spec.weight_column}' (sum={dr_weight_sum})."
        )

    train_df, val_df = train_test_split(data_dr, test_size=test_size, random_state=random_state)

    train_pre = _apply_config_mask(train_df, 'mask_preselection_loose', masks_config)
    val_pre = _apply_config_mask(val_df, 'mask_preselection_loose', masks_config)

    train_ar = _apply_config_mask(train_pre, spec.ar_like_mask_name, masks_config)
    val_ar = _apply_config_mask(val_pre, spec.ar_like_mask_name, masks_config)
    train_sr = _apply_config_mask(train_pre, spec.sr_like_mask_name, masks_config)
    val_sr = _apply_config_mask(val_pre, spec.sr_like_mask_name, masks_config)

    logger.info(
        "%s cutflow: train_pre=(%d, %.6f) val_pre=(%d, %.6f) %s_train=(%d, %.6f) %s_val=(%d, %.6f) %s_train=(%d, %.6f) %s_val=(%d, %.6f)",
        spec.name,
        *_count_and_weight(train_pre, spec.weight_column),
        *_count_and_weight(val_pre, spec.weight_column),
        spec.ar_region_name,
        *_count_and_weight(train_ar, spec.weight_column),
        spec.ar_region_name,
        *_count_and_weight(val_ar, spec.weight_column),
        spec.sr_region_name,
        *_count_and_weight(train_sr, spec.weight_column),
        spec.sr_region_name,
        *_count_and_weight(val_sr, spec.weight_column),
    )

    training_samples = [
        (f"{spec.ar_region_name}_train", train_ar),
        (f"{spec.ar_region_name}_val", val_ar),
        (f"{spec.sr_region_name}_train", train_sr),
        (f"{spec.sr_region_name}_val", val_sr),
    ]
    for sample_name, sample_df in training_samples:
        n_events, sum_weights = _count_and_weight(sample_df, spec.weight_column)
        logger.info(
            "%s sample %s: events=%d, weight_sum=%.6f",
            spec.name,
            sample_name,
            n_events,
            sum_weights,
        )

    numerator = _weight_sum(train_ar, spec.weight_column) + _weight_sum(val_ar, spec.weight_column)
    denominator = _weight_sum(train_sr, spec.weight_column) + _weight_sum(val_sr, spec.weight_column)
    if not np.isfinite(denominator) or denominator == 0.0:
        raise ValueError(
            f"{spec.name}: {spec.sr_region_name} weight sum is zero or invalid. "
            f"Counts train/val = ({len(train_sr)}, {len(val_sr)}), weight sums train/val = "
            f"({_weight_sum(train_sr, spec.weight_column):.6f}, {_weight_sum(val_sr, spec.weight_column):.6f})."
        )

    weight_corr_factor = numerator / denominator
    return {
        spec.ar_region_name: (train_ar, val_ar),
        spec.sr_region_name: (train_sr, val_sr),
    }, weight_corr_factor, data_dr


def build_dataloaders(
    train_df,
    val_df,
    spec: ProcessTrainingSpec,
    config,
    region: str,
    weight_corr_factor: float,
):
    input_variables = ['njets'] + list(variables)
    train_data = spec.data_getter(train_df, input_variables).to_torch(device=None)
    val_data = spec.data_getter(val_df, input_variables).to_torch(device=None)

    x_train = train_data.X
    x_val = val_data.X
    weights_train = train_data.weights
    weights_val = val_data.weights

    weights_train = weights_train / t.sum(weights_train)
    weights_val = weights_val / t.sum(weights_val)

    train_loader = DataLoader(
        TensorDataset(x_train, weights_train),
        batch_size=config.bsize_train,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, weights_val),
        batch_size=config.bsize_val,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    return train_data, val_data, train_loader, val_loader


def save_training_artifacts(checkpoint, log_rows, config, spec: ProcessTrainingSpec, region: str):
    region_dir = Path(spec.output_root) / region
    latest_dir = region_dir / 'latest'
    region_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    t.save(checkpoint, region_dir / 'model_checkpoint.pth')
    t.save(checkpoint, latest_dir / 'model_checkpoint.pth')

    pd.DataFrame(log_rows).to_pickle(str(region_dir / 'training_logs.pkl'))
    pd.DataFrame(log_rows).to_pickle(str(latest_dir / 'training_logs.pkl'))

    # Persist as plain YAML mapping (safe_load-compatible), not a Python object tag.
    cfg = vars(config)
    config_payload = {
        'training': {
            'bsize_train': int(cfg['bsize_train']),
            'bsize_val': int(cfg['bsize_val']),
            'bsize_test': int(cfg['bsize_test']),
            'grad_clip': float(cfg['grad_clip']),
            'n_epochs': int(cfg['n_epochs']),
            'use_amp': bool(cfg['use_amp']),
            's_scale_max': float(cfg['s_scale_max']),
        },
        'model': {
            'n_layers': int(cfg['n_layers']),
            'hidden_dims': int(cfg['hidden_dims']),
            's_scale': float(cfg['s_scale']),
            'use_cut_preprocessing': bool(cfg.get('use_cut_preprocessing', True)),
            'cut_preprocessing_index': list(cfg.get('cut_preprocessing_index', [0, 1])),
            'cut_preprocessing_thresholds': list(cfg.get('cut_preprocessing_thresholds', [33.0, 30.0])),
            'cut_preprocessing_epsilon': float(cfg.get('cut_preprocessing_epsilon', 1e-6)),
            'use_tail_preprocessing': bool(cfg.get('use_tail_preprocessing', False)),
            'tail_preprocessing_index': cfg.get('tail_preprocessing_index', 2),
            'tail_preprocessing_type': cfg.get('tail_preprocessing_type', 'asinh'),
            'tail_preprocessing_center': cfg.get('tail_preprocessing_center', 0.0),
            'tail_preprocessing_scale': cfg.get('tail_preprocessing_scale', 1.0),
            'tail_preprocessing_epsilon': float(cfg.get('tail_preprocessing_epsilon', 1e-6)),
        },
        'optimizer': {
            'lr': float(cfg['lr']),
            'weight_decay': float(cfg['weight_decay']),
            'eps': float(cfg['eps']),
        },
        'scheduler': {
            'step_size': int(cfg['scheduler_step_size']),
            'gamma': float(cfg['scheduler_gamma']),
            'factor': float(cfg['scheduler_factor']),
            'patience': int(cfg['scheduler_patience']),
            'threshold': float(cfg['scheduler_threshold']),
            'cooldown': int(cfg['scheduler_cooldown']),
            'min_lr': float(cfg['scheduler_min_lr']),
            'eps': float(cfg['scheduler_eps']),
        },
    }
    with open(region_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(config_payload, f, sort_keys=False)


# ----- training -----

def train_region(spec: ProcessTrainingSpec, region: str, train_df, val_df, weight_corr_factor, config, device):
    logger.info("Starting %s training for %s", region, spec.name)
    logger.info("%s %s samples: train=%d, val=%d", spec.name, region, len(train_df), len(val_df))

    train_data, val_data, train_loader, val_loader = build_dataloaders(
        train_df,
        val_df,
        spec,
        config,
        region,
        weight_corr_factor,
    )

    dim = len(variables)
    uses_njets_context = True
    model = build_conditional_nf(config, dim, shift=None, scale=None, device=device)
    schema = 'conditional_nf_v1'

    shift, scale, valid_fraction = _compute_preprocessed_scaler_stats(
        model,
        train_data.X,
        uses_njets_context=uses_njets_context,
    )
    _initialize_model_scaler(model, shift, scale)
    logger.info(
        "%s %s scaler initialized on preprocessed features (valid fraction: %.4f)",
        spec.name,
        region,
        valid_fraction,
    )

    optimizer = t.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=config.eps,
    )
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        threshold=config.scheduler_threshold,
        threshold_mode='rel',
        cooldown=config.scheduler_cooldown,
        min_lr=config.scheduler_min_lr,
        eps=config.scheduler_eps,
    )
    use_amp = (device.type == 'cuda') and bool(config.use_amp)
    scaler = t.amp.GradScaler('cuda', enabled=use_amp)

    best_val_nll = float('inf')
    counter = 0
    log_rows = []
    checkpoint = None

    with log.training_dashboard() as dash:
        for epoch in range(1, config.n_epochs + 1):
            epoch_start = time.time()
            model.train()
            train_loss_sum = 0.0
            train_weight_sum = 0.0

            for xb, wb in train_loader:
                xb = xb.to(device, non_blocking=True)
                wb = wb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with t.amp.autocast('cuda', enabled=use_amp):
                    log_px = model(xb).reshape(-1)
                    loss = (-(log_px) * wb).sum()

                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += loss.item()
                train_weight_sum += wb.sum().item()

            avg_train_opt_nll = train_loss_sum / max(train_weight_sum, 1e-12)
            avg_train_nll = evaluate_loader(model, train_loader, device)
            avg_val_nll = evaluate_loader(model, val_loader, device)

            scheduler.step(avg_val_nll)
            epoch_time = time.time() - epoch_start
            current_lr = scheduler.get_last_lr()[0]

            log_rows.append({
                'epoch': epoch,
                'train_loss': avg_train_nll,
                'train_loss_optim': avg_train_opt_nll,
                'val_loss': avg_val_nll,
                'lr': current_lr,
                'time_s': epoch_time,
                'type': 'epoch',
            })

            if avg_val_nll < best_val_nll:
                best_val_nll = avg_val_nll
                counter = 0
                checkpoint = {
                    'optimizer_state_dict': optimizer.state_dict(),
                    'variables': list(variables),
                    'schema': schema,
                    'training_model': TRAINING_MODEL_CONDITIONAL,
                }
                checkpoint['model_state_dict'] = model.state_dict()
            else:
                counter += 1

            dash.update(
                epoch=epoch,
                train_loss=np.round(avg_train_nll, 6),
                val_loss=np.round(avg_val_nll, 6),
                lr=current_lr,
                region=f"{spec.name} {region}",
            )

            if counter >= PATIENCE:
                logger.info("Early stopping triggered for %s %s.", spec.name, region)
                break

    if checkpoint is None:
        raise RuntimeError(f"No checkpoint was created for {spec.name} {region}.")

    save_training_artifacts(checkpoint, log_rows, config, spec, region)
    logger.info("Saved %s training artifacts for %s", spec.name, region)


def train_process(
    spec: ProcessTrainingSpec,
    data_complete,
    config,
    device,
    test_size: float,
    random_state: int,
    masks_config: dict[str, list[str]],
):
    logger.info("Preparing training samples for %s", spec.name)

    region_samples, weight_corr_factor, data_dr = prepare_region_samples(
        data_complete,
        spec,
        test_size=test_size,
        random_state=random_state,
        masks_config=masks_config,
    )
    
    logger.info(
        "%s DR selection contains %d events; weight correction factor %.6f",
        spec.name,
        len(data_dr),
        weight_corr_factor,
    )

    for region in [spec.ar_region_name, spec.sr_region_name]:
        train_df, val_df = region_samples[region]
        train_region(spec, region, train_df, val_df, weight_corr_factor, config, device)


# ----- main -----

def main():
    args = Args().parse_args()

    t.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    config_path = CONFIG_DIR / 'config_NF.yaml'
    config = RealNVP_config.from_yaml(config_path)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)
    logger.info("Training model mode: %s", TRAINING_MODEL_CONDITIONAL)

    tail_variables = variables[4:]
    training_variables_name = f"vars{len(variables)}_{'_'.join(tail_variables)}" if tail_variables else f"vars{len(variables)}_none"
    model_root_dir = Path(args.output_root_base) / f"training_{training_variables_name}"
    logger.info("Model output root: %s", model_root_dir)
    masks_config = load_masks_config(MASKS_CONFIG_PATH)

    data_complete = pd.read_feather(DATA_DIR / 'data_complete.feather')
    logger.info("Loaded %d total events", len(data_complete))

    process_specs: list[ProcessTrainingSpec] = []
    if args.training_process in ('dr_like', 'both'):
        process_specs.append(
            ProcessTrainingSpec(
                name='Wjets_DR',
                process_id=1,
                region_sign_column='OS',
                weight_column='weight',
                output_root=str(model_root_dir / 'Wjets' / 'DR'),
                dr_mask_name='mask_DR_wjets',
                ar_like_mask_name='AR_like',
                sr_like_mask_name='SR_like',
                ar_region_name='AR-like',
                sr_region_name='SR-like',
                data_getter=get_my_data,
            )
        )

    if args.training_process in ('antidr', 'both'):
        process_specs.append(
            ProcessTrainingSpec(
                name='Wjets_antiDR',
                process_id=1,
                region_sign_column='OS',
                weight_column='weight',
                output_root=str(model_root_dir / 'Wjets' / 'antiDR'),
                dr_mask_name='mask_antiDR_wjets',
                ar_like_mask_name='AR',
                sr_like_mask_name='SR',
                ar_region_name='AR',
                sr_region_name='SR',
                data_getter=get_my_data,
            )
        )

    for spec in process_specs:
        logger.info("Launching %s training in mode %s", spec.name, TRAINING_MODEL_CONDITIONAL)
        train_process(
            spec,
            data_complete,
            config,
            device,
            test_size=args.test_size,
            random_state=args.random_state,
            masks_config=masks_config,
        )

    logger.info("Completed all njets trainings")


if __name__ == '__main__':
    main()
