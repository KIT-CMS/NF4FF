import logging
import random
import time
import hashlib
import re
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

from classes.Collection import get_my_data_qcd, get_my_data_wjets
from classes.Dataclasses import RealNVP_config
from classes.NeuralNetworks import ConditionalRealNVP, GroupedNFRouter, RealNVP
from CustomLogging import LogContext, setup_logging


SEED = 42


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

with open('../configs/training_variables.yaml', 'r') as f:
    variables = yaml.safe_load(f)['variables']

logger = setup_logging(logger=logging.getLogger(__name__))
log = LogContext(logger)

PATIENCE = 30

TRAINING_MODEL_GROUPED = 'grouped_njets_split'
TRAINING_MODEL_SINGLE = 'single_nf'
TRAINING_MODEL_CONDITIONAL = 'conditional_nf'

MODE_DIR_BY_TRAINING_MODEL = {
    TRAINING_MODEL_GROUPED: 'split_njets_0_1_ge2',
    TRAINING_MODEL_SINGLE: 'no_njets_split',
    TRAINING_MODEL_CONDITIONAL: 'conditional_njets_input',
}


class Args(Tap):

    split_njets: bool = False  # Deprecated compatibility flag; maps to grouped mode when used alone.
    training_model: Literal['grouped_njets_split', 'single_nf', 'conditional_nf'] = TRAINING_MODEL_CONDITIONAL  # Training mode: grouped split, single inclusive NF, or conditional NF with njets input.
    output_root_base: str = 'Training_results_new'  # Base directory where training folders are written.
    test_size: float = 0.25  # Validation fraction for the train/validation split.
    random_state: int = SEED  # Random seed used for train/validation splitting.

    def configure(self) -> None:
        self.add_argument('--split_njets', action='store_true')


def resolve_training_model(args: Args) -> str:
    # Backward compatibility with the old --split_njets flag.
    if args.split_njets and args.training_model == TRAINING_MODEL_SINGLE:
        logger.warning(
            "--split_njets is deprecated. Mapping to --training_model=%s",
            TRAINING_MODEL_GROUPED,
        )
        return TRAINING_MODEL_GROUPED

    if args.split_njets and args.training_model != TRAINING_MODEL_GROUPED:
        logger.warning("Ignoring --split_njets because --training_model=%s was provided.", args.training_model)

    return args.training_model


@dataclass(frozen=True)
class ProcessTrainingSpec:
    name: str
    region_sign_column: str
    weight_column: str
    output_root: str
    dr_mask: Callable[[pd.DataFrame], pd.DataFrame]
    data_getter: Callable


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


# ----- shared helpers -----

def mask_preselection_loose(df):
    mask_eta = (df.eta_1 <= 2.1) & (df.eta_2 <= 2.3)
    mask_pt = (df.pt_1 >= 33) & (df.pt_2 >= 30)
    mask_tau_decay_mode = (
        (df.tau_decaymode_2 == 0)
        | (df.tau_decaymode_2 == 1)
        | (df.tau_decaymode_2 == 10)
        | (df.tau_decaymode_2 == 11)
    )
    return df[mask_eta & mask_pt & mask_tau_decay_mode]


def SR_like(df):
    return df[df.id_tau_vsJet_Tight_2 > 0.5]


def AR_like(df):
    mask = (df.id_tau_vsJet_VLoose_2 > 0.5) & (df.id_tau_vsJet_Tight_2 < 0.5)
    return df[mask]


def mask_DR_wjets(df):
    mask = (
        (df.id_tau_vsJet_VLoose_2 > 0.5)
        & (df.nbtag == 0)
        & (df.iso_1 > 0.0)
        & (df.iso_1 < 0.15)
        & (df.extramuon_veto < 0.5)
        & (df.extraelec_veto < 0.5)
        & (df.mt_1 > 70)
    )
    return df[mask].copy()


def mask_DR_qcd(df):
    mask = (
        (df.id_tau_vsJet_VLoose_2 > 0.5)
        & (df.q_1 * df.q_2 > 0)
        & (df.iso_1 > 0.02)
        & (df.iso_1 < 0.15)
        & (df.extramuon_veto < 0.5)
        & (df.extraelec_veto < 0.5)
        & (df.mt_1 < 50)
    )
    return df[mask].copy()


def evaluate_loader(model, loader, device):
    model.eval()
    loss_sum = 0.0
    weight_sum = 0.0

    with t.no_grad():
        for Xb, Wb in loader:
            Xb = Xb.to(device, non_blocking=True)
            Wb = Wb.to(device, non_blocking=True)

            with t.amp.autocast('cuda', enabled=False):
                log_px = model(Xb).reshape(-1)
                loss = (-(log_px) * Wb).sum()

            loss_sum += loss.item()
            weight_sum += Wb.sum().item()

    return loss_sum / max(weight_sum, 1e-12)


def build_grouped_router(config, dim, shift, scale, device):
    model_0 = RealNVP(
        dim=dim,
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
    model_1 = RealNVP(
        dim=dim,
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
    model_2 = RealNVP(
        dim=dim,
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
        model_0.initialize_scaler(shift, scale)
        model_1.initialize_scaler(shift, scale)
        model_2.initialize_scaler(shift, scale)

    router = GroupedNFRouter().to(device)
    router._fallback_payload = model_2
    router._wrapped_delegate = model_2

    router.models.append(model_0)
    router.models.append(model_1)
    router.models.append(model_2)
    router._logic_pipeline = [
        ([(0, (0,))], model_0),
        ([(0, (1,))], model_1),
        ([(0, (2, 1000))], model_2),
    ]
    return router


def build_single_nf(config, dim, shift, scale, device):
    model = RealNVP(
        dim=dim,
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


def _initialize_model_scaler(model, shift: t.Tensor, scale: t.Tensor, training_model: str):
    if training_model == TRAINING_MODEL_GROUPED:
        for sub_model in model.models:
            sub_model.initialize_scaler(shift, scale)
    else:
        model.initialize_scaler(shift, scale)


def prepare_region_samples(data_complete, spec: ProcessTrainingSpec, test_size: float, random_state: int):
    data_dr = spec.dr_mask(data_complete)
    data_dr = data_dr[(data_dr.process == 0) & (data_dr[spec.region_sign_column] == True)].reset_index(drop=True)

    train_df, val_df = train_test_split(data_dr, test_size=test_size, random_state=random_state)

    train_ar = mask_preselection_loose(AR_like(train_df))
    val_ar = mask_preselection_loose(AR_like(val_df))
    train_sr = mask_preselection_loose(SR_like(train_df))
    val_sr = mask_preselection_loose(SR_like(val_df))

    numerator = pd.concat([train_ar[spec.weight_column], val_ar[spec.weight_column]]).sum()
    denominator = pd.concat([train_sr[spec.weight_column], val_sr[spec.weight_column]]).sum()
    if denominator == 0:
        raise ValueError(f"{spec.name}: SR-like weight sum is zero.")

    weight_corr_factor = numerator / denominator
    return {
        'AR-like': (train_ar, val_ar),
        'SR-like': (train_sr, val_sr),
    }, weight_corr_factor, data_dr


def build_dataloaders(
    train_df,
    val_df,
    spec: ProcessTrainingSpec,
    config,
    region: str,
    weight_corr_factor: float,
    training_model: str,
):
    include_njets = training_model in (TRAINING_MODEL_GROUPED, TRAINING_MODEL_CONDITIONAL)
    input_variables = (['njets'] + variables) if include_njets else list(variables)
    train_data = spec.data_getter(train_df, input_variables).to_torch(device=None)
    val_data = spec.data_getter(val_df, input_variables).to_torch(device=None)

    x_train = train_data.X
    x_val = val_data.X
    weights_train = train_data.weights
    weights_val = val_data.weights

    weights_train = weights_train / t.sum(weights_train)
    weights_val = weights_val / t.sum(weights_val)

    if region == 'SR-like':
        weights_train = weights_train * weight_corr_factor
        weights_val = weights_val * weight_corr_factor

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

    with open(region_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)


# ----- training -----

def train_region(spec: ProcessTrainingSpec, region: str, train_df, val_df, weight_corr_factor, config, device, training_model: str):
    logger.info("Starting %s training for %s", region, spec.name)
    logger.info("%s %s samples: train=%d, val=%d", spec.name, region, len(train_df), len(val_df))

    train_data, val_data, train_loader, val_loader = build_dataloaders(
        train_df,
        val_df,
        spec,
        config,
        region,
        weight_corr_factor,
        training_model,
    )

    dim = len(variables)
    uses_njets_context = training_model in (TRAINING_MODEL_GROUPED, TRAINING_MODEL_CONDITIONAL)

    if training_model == TRAINING_MODEL_GROUPED:
        model = build_grouped_router(config, dim, shift=None, scale=None, device=device)
        schema = 'grouped_nf_router_v1'
    elif training_model == TRAINING_MODEL_CONDITIONAL:
        model = build_conditional_nf(config, dim, shift=None, scale=None, device=device)
        schema = 'conditional_nf_v1'
    else:
        model = build_single_nf(config, dim, shift=None, scale=None, device=device)
        schema = 'single_nf_v1'

    shift, scale, valid_fraction = _compute_preprocessed_scaler_stats(
        model,
        train_data.X,
        uses_njets_context=uses_njets_context,
    )
    _initialize_model_scaler(model, shift, scale, training_model)
    logger.info(
        "%s %s scaler initialized on preprocessed features (valid fraction: %.4f)",
        spec.name,
        region,
        valid_fraction,
    )

    optimizer = t.optim.AdamW(model.parameters(), lr=config.lr)
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
    scaler = t.amp.GradScaler('cuda', enabled=config.use_amp)

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
                with t.amp.autocast('cuda', enabled=False):
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
                    'training_model': training_model,
                }
                if training_model == TRAINING_MODEL_GROUPED:
                    checkpoint['router_state_dict'] = model.state_dict()
                else:
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


def train_process(spec: ProcessTrainingSpec, data_complete, config, device, training_model: str, test_size: float, random_state: int):
    logger.info("Preparing training samples for %s", spec.name)
    region_samples, weight_corr_factor, data_dr = prepare_region_samples(data_complete, spec, test_size=test_size, random_state=random_state)
    logger.info(
        "%s DR selection contains %d events; weight correction factor %.6f",
        spec.name,
        len(data_dr),
        weight_corr_factor,
    )

    for region in ['AR-like', 'SR-like']:
        train_df, val_df = region_samples[region]
        train_region(spec, region, train_df, val_df, weight_corr_factor, config, device, training_model=training_model)


# ----- main -----

def main():
    args = Args().parse_args()
    training_model = resolve_training_model(args)

    t.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    config_path = '../configs/config_NF.yaml'
    config = RealNVP_config.from_yaml(config_path)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)
    logger.info("Training model mode: %s", training_model)

    mode_dir = MODE_DIR_BY_TRAINING_MODEL[training_model]
    training_variables_tag = build_training_variables_tag(variables)
    model_root_dir = Path(args.output_root_base) / mode_dir / f"training_{training_variables_tag}"
    logger.info("Model output root: %s", model_root_dir)

    data_complete = pd.read_feather('../../data/data_complete.feather')
    logger.info("Loaded %d total events", len(data_complete))

    process_specs = [
        ProcessTrainingSpec(
            name='Wjets',
            region_sign_column='OS',
            weight_column='weight_wjets',
            output_root=str(model_root_dir / 'Wjets' / 'all'),
            dr_mask=mask_DR_wjets,
            data_getter=get_my_data_wjets,
        ),
        ProcessTrainingSpec(
            name='QCD',
            region_sign_column='SS',
            weight_column='weight_qcd',
            output_root=str(model_root_dir / 'QCD' / 'all'),
            dr_mask=mask_DR_qcd,
            data_getter=get_my_data_qcd,
        ),
    ]

    for spec in process_specs:
        logger.info("Launching %s training in mode %s", spec.name, training_model)
        train_process(
            spec,
            data_complete,
            config,
            device,
            training_model=training_model,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    logger.info("Completed all njets trainings")


if __name__ == '__main__':
    main()
