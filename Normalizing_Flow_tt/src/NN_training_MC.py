import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from tap import Tap
from torch.utils.data import DataLoader, TensorDataset

from classes.Collection import get_my_data_wjets
from classes.Dataclasses import Config
from classes.NeuralNetworks import BinaryClassifier
from CustomLogging import LogContext, setup_logging


SEED = 42
PATIENCE = 30


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)
variables = [
    "pt_1","pt_2","eta_1","eta_2","jpt_1","jpt_2","jeta_1","jeta_2",
    "m_fastmtt","pt_fastmtt","met","njets","mt_tot","m_vis",
    "pt_tt","pt_vis","mjj","pt_dijet","pt_ttjj","deltaEta_jj","deltaR_jj",
    "deltaR_ditaupair","deltaR_1j1","deltaR_1j2",
    "deltaR_2j1","deltaR_2j2","deltaR_12j1","deltaR_12j2","deltaEta_1j1",
    "deltaEta_1j2","deltaEta_2j1","deltaEta_2j2","deltaEta_12j1","deltaEta_12j2", 'tau_decaymode_1', 'tau_decaymode_2'
]

logger = setup_logging(logger=logging.getLogger(__name__))
log = LogContext(logger)


class Args(Tap):
    output_root_base: str = 'Training_results_new'
    test_size: float = 0.25
    random_state: int = SEED


@dataclass(frozen=True)
class ProcessTrainingSpec:
    name: str
    weight_column: str
    output_root: str
    dataset_builder: Callable[[pd.DataFrame], pd.DataFrame]
    data_getter: Callable


def build_training_model_tag(
    hidden_layers: int,
    hidden_dim: int,
    dropout: float,
) -> str:
    dropout_token = str(dropout).replace(".", "p")
    return f"hl{hidden_layers}_hd{hidden_dim}_do{dropout_token}"


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

def mask_antiDR_wjets(df):
    mask = (
        (df.id_tau_vsJet_VLoose_2 > 0.5)
        & (df.nbtag == 0)
        & (df.iso_1 > 0.0)
        & (df.iso_1 < 0.15)
        & (df.extramuon_veto < 0.5)
        & (df.extraelec_veto < 0.5)
        & (df.mt_1 <= 70)
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


def _build_pt2_bins(values: np.ndarray, n_bins: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(0.0, 200.0, n_bins + 1, dtype=np.float32)

    bins = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)).astype(np.float32)
    bins = np.unique(bins)
    if bins.size < 2:
        center = float(values[0])
        bins = np.linspace(max(0.0, center - 1.0), center + 1.0, n_bins + 1, dtype=np.float32)
    return bins


def build_wjets_mc_dataset(data_complete: pd.DataFrame) -> pd.DataFrame:
    data_dr = mask_DR_wjets(data_complete)
    dataset = data_dr[(data_dr.process == 1) & (data_dr.OS == True)].copy()
    dataset['weight_wjets'] = dataset['weight'].to_numpy(dtype=np.float32)
    return dataset.reset_index(drop=True)


def build_wjets_data_dr_dataset(data_complete: pd.DataFrame) -> pd.DataFrame:
    data_region = mask_DR_wjets(data_complete)
    dataset = data_region[(data_region.process == 0) & (data_region.OS == True)].copy()
    dataset['weight_wjets'] = dataset['weight'].to_numpy(dtype=np.float32)
    return dataset.reset_index(drop=True)


def build_wjets_data_antidr_dataset(data_complete: pd.DataFrame) -> pd.DataFrame:
    data_region = mask_antiDR_wjets(data_complete)
    dataset = data_region[(data_region.process == 0) & (data_region.OS == True)].copy()
    dataset['weight_wjets'] = dataset['weight'].to_numpy(dtype=np.float32)
    return dataset.reset_index(drop=True)


def sanitize_weights(weights: t.Tensor, process_name: str, split_name: str) -> t.Tensor:
    negative_mask = weights < 0
    n_negative = int(negative_mask.sum().item())
    if n_negative > 0:
        logger.warning(
            "%s %s: found %d negative event weights; clipping them to 0 for BCE stability.",
            process_name,
            split_name,
            n_negative,
        )
        weights = weights.clone()
        weights[negative_mask] = 0.0

    w_sum = t.sum(weights)
    if float(w_sum.item()) <= 0.0:
        raise ValueError(f"{process_name} {split_name}: non-positive total weight after sanitization.")

    return weights / w_sum


def smooth_binary_targets(targets: t.Tensor, label_smoothing: float) -> t.Tensor:
    s = float(label_smoothing)
    if s <= 0.0:
        return targets
    s = min(s, 0.499)
    return targets * (1.0 - s) + 0.5 * s


@t.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: t.device,
    label_smoothing: float = 0.0,
) -> float:
    model.eval()
    loss_sum = 0.0
    weight_sum = 0.0

    for xb, yb, wb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        wb = wb.to(device, non_blocking=True)

        prob_sr = model(xb).reshape(-1)
        yb_smooth = smooth_binary_targets(yb, label_smoothing)
        bce = F.binary_cross_entropy(prob_sr, yb_smooth, reduction='none')
        loss_sum += (bce * wb).sum().item()
        weight_sum += wb.sum().item()

    return loss_sum / max(weight_sum, 1e-12)


@t.no_grad()
def density_ratio_sr_over_ar(
    model: nn.Module,
    x: t.Tensor,
    prior_ar_over_sr: float,
    eps: float = 1e-7,
) -> t.Tensor:
    """
    Convert classifier output P(SR-like|x) into density ratio p(x|SR-like)/p(x|AR-like):

        ratio(x) = [P(SR|x) / (1 - P(SR|x))] * [P(AR) / P(SR)]

    where priors are weighted class priors from the training sample.
    """
    prob_sr = t.clamp(model(x).reshape(-1), min=eps, max=1.0 - eps)
    odds = prob_sr / (1.0 - prob_sr)
    return odds * prior_ar_over_sr


def prepare_process_samples(
    data_complete: pd.DataFrame,
    spec: ProcessTrainingSpec,
    test_size: float,
    random_state: int,
):
    data_dr = spec.dataset_builder(data_complete).reset_index(drop=True)

    if len(data_dr) == 0:
        raise ValueError(f"{spec.name}: DR selection is empty.")

    train_df, val_df = train_test_split(data_dr, test_size=test_size, random_state=random_state)

    train_ar = mask_preselection_loose(AR_like(train_df)).copy()
    val_ar = mask_preselection_loose(AR_like(val_df)).copy()
    train_sr = mask_preselection_loose(SR_like(train_df)).copy()
    val_sr = mask_preselection_loose(SR_like(val_df)).copy()

    train_ar['target_sr'] = 0.0
    val_ar['target_sr'] = 0.0
    train_sr['target_sr'] = 1.0
    val_sr['target_sr'] = 1.0

    train_mix = pd.concat([train_ar, train_sr], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_mix = pd.concat([val_ar, val_sr], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    sr_weight = pd.concat([train_sr[spec.weight_column], val_sr[spec.weight_column]], axis=0).sum()
    ar_weight = pd.concat([train_ar[spec.weight_column], val_ar[spec.weight_column]], axis=0).sum()

    if sr_weight <= 0 or ar_weight <= 0:
        raise ValueError(
            f"{spec.name}: invalid weighted priors (SR={sr_weight}, AR={ar_weight})."
        )

    prior_ar_over_sr = float(ar_weight / sr_weight)

    return {
        'train': train_mix,
        'val': val_mix,
        'n_train_ar': int(len(train_ar)),
        'n_train_sr': int(len(train_sr)),
        'n_val_ar': int(len(val_ar)),
        'n_val_sr': int(len(val_sr)),
        'sr_weight': float(sr_weight),
        'ar_weight': float(ar_weight),
        'prior_ar_over_sr': prior_ar_over_sr,
        'data_dr_size': int(len(data_dr)),
    }


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    spec: ProcessTrainingSpec,
    feature_columns: list[str],
    config: Config,
):
    train_data = spec.data_getter(train_df, feature_columns).to_torch(device=None)
    val_data = spec.data_getter(val_df, feature_columns).to_torch(device=None)

    x_train = train_data.X
    x_val = val_data.X

    y_train = t.from_numpy(train_df['target_sr'].to_numpy(dtype=np.float32))
    y_val = t.from_numpy(val_df['target_sr'].to_numpy(dtype=np.float32))

    w_train = sanitize_weights(train_data.weights, spec.name, 'train')
    w_val = sanitize_weights(val_data.weights, spec.name, 'val')

    train_loader = DataLoader(
        TensorDataset(x_train, y_train, w_train),
        batch_size=config.bsize_train,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val, w_val),
        batch_size=config.bsize_val,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    return train_data, val_data, train_loader, val_loader


def save_training_artifacts(
    checkpoint: dict,
    log_rows: list[dict],
    config: Config,
    spec: ProcessTrainingSpec,
):
    process_dir = Path(spec.output_root) / 'SR_AR_classifier'
    latest_dir = process_dir / 'latest'
    process_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    t.save(checkpoint, process_dir / 'model_checkpoint.pth')
    t.save(checkpoint, latest_dir / 'model_checkpoint.pth')

    pd.DataFrame(log_rows).to_pickle(str(process_dir / 'training_logs.pkl'))
    pd.DataFrame(log_rows).to_pickle(str(latest_dir / 'training_logs.pkl'))

    with open(process_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(checkpoint['training_config'], f, sort_keys=False)


# ----- training -----

def train_process(
    spec: ProcessTrainingSpec,
    data_complete: pd.DataFrame,
    config: Config,
    device: t.device,
    feature_columns: list[str],
    hidden_dim: int,
    hidden_layers: int,
    dropout: float,
    test_size: float,
    random_state: int,
):
    logger.info("Preparing %s samples", spec.name)
    label_smoothing = float(max(0.0, getattr(config, 'label_smoothing', 0.0)))
    prep = prepare_process_samples(
        data_complete,
        spec,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info(
        "%s region events=%d | train(AR=%d, SR=%d) | val(AR=%d, SR=%d) | prior AR/SR=%.6f",
        spec.name,
        prep['data_dr_size'],
        prep['n_train_ar'],
        prep['n_train_sr'],
        prep['n_val_ar'],
        prep['n_val_sr'],
        prep['prior_ar_over_sr'],
    )

    train_data, _, train_loader, val_loader = build_dataloaders(
        prep['train'],
        prep['val'],
        spec,
        feature_columns,
        config,
    )

    input_dim = len(feature_columns)
    model = BinaryClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        p=dropout,
        hidden_layers=hidden_layers,
    ).to(device)

    shift = train_data.X.mean(dim=0)
    scale = train_data.X.std(dim=0, unbiased=False).clamp_min(1e-12)
    model.initialize_scaler(shift=shift, scale=scale)

    optimizer = t.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
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
    scaler = t.amp.GradScaler('cuda', enabled=config.use_amp)

    best_val_loss = float('inf')
    counter = 0
    checkpoint = None
    log_rows = []

    with log.training_dashboard() as dash:
        for epoch in range(1, config.n_epochs + 1):
            epoch_start = time.time()
            model.train()

            train_loss_sum = 0.0
            train_weight_sum = 0.0

            for xb, yb, wb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                wb = wb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with t.amp.autocast('cuda', enabled=False):
                    prob_sr = model(xb).reshape(-1)
                    yb_smooth = smooth_binary_targets(yb, label_smoothing)
                    bce = F.binary_cross_entropy(prob_sr, yb_smooth, reduction='none')
                    loss = (bce * wb).sum()

                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += loss.item()
                train_weight_sum += wb.sum().item()

            avg_train_opt_loss = train_loss_sum / max(train_weight_sum, 1e-12)
            avg_train_loss = evaluate_loader(model, train_loader, device, label_smoothing=label_smoothing)
            avg_val_loss = evaluate_loader(model, val_loader, device, label_smoothing=label_smoothing)

            scheduler.step(avg_val_loss)
            epoch_time = time.time() - epoch_start
            current_lr = scheduler.get_last_lr()[0]

            log_rows.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'train_loss_optim': avg_train_opt_loss,
                'val_loss': avg_val_loss,
                'lr': current_lr,
                'time_s': epoch_time,
                'type': 'epoch',
            })

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'variables': list(variables),
                    'feature_columns': list(feature_columns),
                    'schema': 'binary_classifier_sr_ar_ratio_v1',
                    'process': spec.name,
                    'prior_sr_weight': prep['sr_weight'],
                    'prior_ar_weight': prep['ar_weight'],
                    'prior_ar_over_sr': prep['prior_ar_over_sr'],
                    'training_config': {
                        'base_config_path': '../configs/config_NN.yaml',
                        'dataset_type': 'wjets_corrections_data_regions',
                        'seed': SEED,
                        'test_size': test_size,
                        'random_state': random_state,
                        'hidden_dim': hidden_dim,
                        'hidden_layers': hidden_layers,
                        'dropout': dropout,
                        'bsize_train': config.bsize_train,
                        'bsize_val': config.bsize_val,
                        'n_epochs': config.n_epochs,
                        'lr': config.lr,
                        'weight_decay': config.weight_decay,
                        'label_smoothing': label_smoothing,
                        'grad_clip': config.grad_clip,
                        'scheduler_factor': config.scheduler_factor,
                        'scheduler_patience': config.scheduler_patience,
                        'scheduler_threshold': config.scheduler_threshold,
                        'scheduler_cooldown': config.scheduler_cooldown,
                        'scheduler_min_lr': config.scheduler_min_lr,
                        'scheduler_eps': config.scheduler_eps,
                    },
                }
            else:
                counter += 1

            dash.update(
                epoch=epoch,
                train_loss=np.round(avg_train_loss, 6),
                val_loss=np.round(avg_val_loss, 6),
                lr=current_lr,
                region=spec.name,
            )

            if counter >= PATIENCE:
                logger.info("Early stopping triggered for %s.", spec.name)
                break

    if checkpoint is None:
        raise RuntimeError(f"No checkpoint was created for {spec.name}.")

    save_training_artifacts(checkpoint, log_rows, config, spec)
    logger.info("Saved classifier artifacts for %s", spec.name)


# ----- main -----

def main():
    args = Args(explicit_bool=True).parse_args()

    t.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    config_path = '../configs/config_NN.yaml'
    config = Config.from_yaml(config_path)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    feature_columns = list(variables)

    training_model_tag = build_training_model_tag(
        hidden_layers=config.n_layers,
        hidden_dim=config.hidden_dims,
        dropout=config.dropout,
    )
    model_root_dir = Path(args.output_root_base) / 'binary_classifier_corrections' / f"training_{training_model_tag}"
    logger.info("Model output root: %s", model_root_dir)

    data_complete = pd.read_feather('../../data/data_complete.feather')
    logger.info("Loaded %d total events", len(data_complete))

    process_specs = [
        ProcessTrainingSpec(
            name='Wjets_DR',
            weight_column='weight_wjets',
            output_root=str(model_root_dir / 'Wjets' / 'DR'),
            dataset_builder=build_wjets_data_dr_dataset,
            data_getter=get_my_data_wjets,
        ),
        ProcessTrainingSpec(
            name='Wjets_antiDR',
            weight_column='weight_wjets',
            output_root=str(model_root_dir / 'Wjets' / 'antiDR'),
            dataset_builder=build_wjets_data_antidr_dataset,
            data_getter=get_my_data_wjets,
        ),
    ]

    for spec in process_specs:
        train_process(
            spec=spec,
            data_complete=data_complete,
            config=config,
            device=device,
            feature_columns=feature_columns,
            hidden_dim=config.hidden_dims,
            hidden_layers=config.n_layers,
            dropout=config.dropout,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    logger.info("Completed Wjets DR and antiDR SR/AR binary-classifier correction training.")


if __name__ == '__main__':
    main()
