"""Enrichment-model training and feature inference."""

import numpy as np
from pathlib import Path
import logging
from core.paths import CONFIG_ROOT, PROJECT_ROOT
import json
import random
import torch as t
import torch.nn as nn
import pandas as pd
from typing import Union, Any, Dict, Tuple, Literal
from core.config import load_config
from data.handling import FeatureRegistry, FeatureStore, load_data, load_data_no_embedding, load_variables
from models.networks import DNN, FoldCombinedDNN, save_model
from data.components import get_class_weights, _same_sign_opposite_sign_split, _collection, _component_collection
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, fields
from logging_utils.context import setup_logging
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from training.enrichment_classifier import (
    QCD_SS_WEIGHT_DYNAMIC_DELTA,
    QCD_SS_WEIGHT_DYNAMIC_DELTA_LAST,
    QCD_SS_WEIGHT_DYNAMIC_MIN_QCD_YIELD,
    QCD_WEIGHT_BINNING,
    QCD_WEIGHT_N_BINS,
    _calculate_scaled_event_weights_generalized,
    build_qcd_weight_bins,
    evaluate_binary_classifier,
    get_my_data,
    predict_probabilities,
    refresh_qcd_weights,
    should_refresh_qcd_weights,
)
from rich.console import Console
from rich.table import Table
from rich.rule import Rule

import time

logger = setup_logging(logger=logging.getLogger(__name__))
console = Console()

# ---------- dataclasses -------------


QCD_WEIGHT_BINNING = 'quantile'
QCD_WEIGHT_N_BINS = 40
QCD_WEIGHT_DYNAMIC_DELTA = 10.0
QCD_WEIGHT_DYNAMIC_DELTA_LAST = 10.0
QCD_WEIGHT_DYNAMIC_MIN_QCD_YIELD = 10.0
QCD_WEIGHT_REFRESH_EVERY = 5
QCD_WEIGHT_REFRESH_UNTIL_EPOCH = 100
QCD_GROUPING_CONFIG_PATH = CONFIG_ROOT / "grouping_enrichment.yaml"
TRAINING_SEED = 42
QCD_EARLY_STOPPING_PATIENCE = 20


def _set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False


def _normalize_groups(groups_raw):
    return tuple(tuple(int(v) for v in group) for group in groups_raw)


def _load_qcd_grouping_config(path: str) -> Dict[str, Any]:
    cfg = load_config(path)
    qcd_cfg = cfg["qcd_reweighting"]
    return {
        "use_grouping": bool(qcd_cfg.get("use_grouping", True)),
        "default_grouping": {
            "index_name": qcd_cfg["default_grouping"]["index_name"],
            "groups": _normalize_groups(qcd_cfg["default_grouping"]["groups"]),
        },
        "groupings": [
            {
                "name": item.get("name", item["index_name"]),
                "index_name": item["index_name"],
                "groups": _normalize_groups(item["groups"]),
            }
            for item in qcd_cfg["groupings"]
        ],
    }


def _validate_grouping_config(training_variables, grouping_cfg):
    known_vars = set(training_variables)
    for item in grouping_cfg["groupings"]:
        name = item["name"]
        index_name = item["index_name"]
        groups = item["groups"]
        if index_name not in known_vars:
            raise ValueError(
                f"Grouping variable '{index_name}' for '{name}' "
                "not found in training variables"
            )
        if not isinstance(groups, tuple) or len(groups) == 0:
            raise ValueError(f"Grouping for '{name}' must be a non-empty tuple of groups")
        for group in groups:
            if len(group) not in (1, 2):
                raise ValueError(
                    f"Invalid group {group} for '{name}'. Each group must be (value,) or (min,max)."
                )
        intervals = [
            (group[0], group[0]) if len(group) == 1 else group
            for group in groups
        ]
        for group_index, (low, high) in enumerate(intervals):
            for other_index in range(group_index + 1, len(intervals)):
                other_low, other_high = intervals[other_index]
                if max(low, other_low) <= min(high, other_high):
                    raise ValueError(
                        f"Overlapping groups for '{name}': "
                        f"{groups[group_index]} and "
                        f"{groups[other_index]}"
                    )


@dataclass
class Config:
    # training
    bsize_train: int     # not used in full-batch (kept for compatibility)
    bsize_val: int       # not used in full-batch
    bsize_test: int      # not used in full-batch
    grad_clip: float
    n_epochs: int
    use_amp: bool
    s_scale_max: float
    patience: int
    # optimizer
    lr: float

    # scheduler
    scheduler_step_size: int
    scheduler_gamma: float
    scheduler_factor: float
    scheduler_patience: int
    scheduler_threshold: float
    scheduler_cooldown: int
    scheduler_min_lr: float
    scheduler_eps: float

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "Config":
        training = cfg["training"]
        optimizer = cfg["optimizer"]
        scheduler = cfg["scheduler"]

        return Config(
            bsize_train=training["bsize_train"],
            bsize_val=training["bsize_val"],
            bsize_test=training["bsize_test"],
            grad_clip=training["grad_clip"],
            n_epochs=training["n_epochs"],
            use_amp=training["use_amp"],
            s_scale_max=training["s_scale_max"],
            patience=training["patience"],
            lr=optimizer["lr"],
            scheduler_step_size=scheduler["step_size"],
            scheduler_gamma=scheduler["gamma"],
            scheduler_factor=scheduler["factor"],
            scheduler_patience=scheduler["patience"],
            scheduler_threshold=scheduler["threshold"],
            scheduler_cooldown=scheduler["cooldown"],
            scheduler_min_lr=scheduler["min_lr"],
            scheduler_eps=scheduler["eps"],
        )




#-------------- helpers --------------


def print_training_header(fold_label):
    console.print(
        Rule(f"[bold cyan]{fold_label}[/bold cyan]")
    )


def print_epoch_summary(
    fold_label,
    cfg,
    epoch,
    n_epochs,
    train_loss,
    val_loss,
    lr,
    epoch_time,
):
    table = Table(
        expand=False,
    )

    table.add_column("Epoch", justify="right")
    table.add_column("Train Loss", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("LR", justify="right")
    table.add_column("Time", justify="right")

    table.add_row(
        f"{epoch}/{n_epochs}",
        f"{train_loss:.6f}",
        f"{val_loss:.6f}",
        f"{lr:.2e}",
        f"{epoch_time:.1f}s",
    )

    console.print(table)


@t.no_grad()
def _evaluate_qcd_classifier(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    weight_sum = 0.0
    correct = 0
    total = 0

    for Xb, yb, wb in loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        wb = wb.to(device, non_blocking=True)

        predictions = model(Xb)
        targets = (yb == 2).float().view(-1, 1)
        weights = wb.float().view(-1, 1)
        loss_per_event = criterion(
            predictions.float().clamp(1e-7, 1 - 1e-7),
            targets,
        )

        loss_sum += (loss_per_event * weights).sum().item()
        weight_sum += weights.sum().item()
        predicted_classes = (predictions >= 0.5).float()
        correct += (predicted_classes.view(-1) == targets.view(-1)).sum().item()
        total += targets.numel()

    return (
        loss_sum / max(weight_sum, 1e-12),
        correct / max(total, 1),
    )


def _write_group_metadata(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))



def _train_fold_model(
        cfg,
        df,
        training_variables,
        group_idx,
        grouping,
        use_grouping,
        device,
        checkpoint_dir,
        fold_label,
        seed: int = TRAINING_SEED,
):

    def _split_collection(dataset, ss_idx, os_idx):
        split_fields = {}
        for field in fields(dataset):
            value = getattr(dataset, field.name)
            if value is None:
                split_fields[field.name] = None
            elif hasattr(value, "ss") and hasattr(value, "os"):
                split_fields[field.name] = type(value)(
                    ss=value.ss[ss_idx] if value.ss is not None else None,
                    os=value.os[os_idx] if value.os is not None else None,
                )
            elif field.name.endswith("_os"):
                split_fields[field.name] = value[os_idx]
            else:
                split_fields[field.name] = value
        return type(dataset)(**split_fields)

    train_idx_ss, val_idx_ss = train_test_split(np.arange(df.X.ss.shape[0]), random_state=seed)
    train_idx_os, val_idx_os = train_test_split(np.arange(df.X.os.shape[0]), random_state=seed)
    train_pt = _split_collection(df, train_idx_ss, train_idx_os).to_torch(device=None)
    val_pt = _split_collection(df, val_idx_ss, val_idx_os).to_torch(device=None)

    X_train = train_pt.X.os

    shift = X_train.mean(dim=0).to(device)
    scale  = X_train.std(dim=0, unbiased=False).clamp_min(1e-12).to(device)

    model = DNN(
        input_nodes = len(training_variables),
        hidden_nodes = (200, 200), 
        output_nodes = 1,
        dropout = 0.15,
        activation = 'ReLU',
        output_activation='Sigmoid',
        input_names = training_variables,
    )

    model.initialize_scaler(shift = shift, scale = scale)

    criterion = nn.BCELoss(reduction='none')                                 
    optimizer = t.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        threshold=cfg.scheduler_threshold,
        threshold_mode='rel',
        cooldown=cfg.scheduler_cooldown,
        min_lr=cfg.scheduler_min_lr,
        eps=cfg.scheduler_eps
    )


    use_amp = (device.type == "cuda") and bool(cfg.use_amp)
    scaler_amp = t.amp.GradScaler('cuda', enabled=use_amp)

    # training loop (full-batch)

    best_val = float('inf')
    counter = 0
    checkpoint = None

    log_rows = []
    logger.info("Starting training for %s", fold_label)




    qcd_mask_os_train = (train_pt.Y.os == 2)
    qcd_mask_os_val = (val_pt.Y.os == 2)

    train_generator = t.Generator().manual_seed(seed)
    val_generator = t.Generator().manual_seed(seed + 1)

    print_training_header(fold_label)

    for epoch in range(1, cfg.n_epochs + 1):

        refresh_qcd = should_refresh_qcd_weights(epoch)

        # ------- update qcd weights (every 5 epochs) ------

        if refresh_qcd:
            model.eval()
            with t.no_grad():
                train_pt = refresh_qcd_weights(
                    dataset=train_pt,
                    model=model,
                    qcd_mask_os_loaded=qcd_mask_os_train,
                    device=device,
                    group_idx=group_idx,
                    grouping=grouping,
                    use_grouping=use_grouping,
                )

                val_pt = refresh_qcd_weights(
                    dataset=val_pt,
                    model=model,
                    qcd_mask_os_loaded=qcd_mask_os_val,
                    device=device,
                    group_idx=group_idx,
                    grouping=grouping,
                    use_grouping=use_grouping,
                )





        X_train = train_pt.X.os
        y_train = train_pt.Y.os
        w_train = train_pt.weights.os

        X_val = val_pt.X.os
        y_val = val_pt.Y.os
        w_val = val_pt.weights.os


        dataset_train = TensorDataset(X_train, y_train, w_train, train_pt.process.os)
        dataset_val   = TensorDataset(X_val,   y_val,   w_val,   val_pt.process.os)


        train_loader = DataLoader(
            dataset_train,
            batch_size = cfg.bsize_train,
            shuffle = True,
            drop_last = False,
            generator=train_generator,
        )

        val_loader = DataLoader(
            dataset_val,
            batch_size = cfg.bsize_val,
            shuffle= True,
            drop_last = False,
            generator=val_generator,
        )

        # ------- train

        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        epoch_start = time.time()

        for Xb, yb, wb, pb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking = True)

            optimizer.zero_grad(set_to_none=True)


            with t.amp.autocast('cuda', enabled=use_amp):

                logits = model(Xb)
                y = (yb == 1).float().view(-1, 1)
                w = wb.float().view(-1, 1)

                # Identify classes
                is_signal = (y == 1).float()
                is_qcd    = (pb == 2).float().view(-1, 1)   # process ID 2 = QCD
                is_bkg    = ((y == 0).float() * (1 - is_qcd))

                # Base BCE loss (per event)
                base_loss = criterion(logits.float().clamp(1e-7, 1 - 1e-7), y)

                # Weighting factors (tune these)
                w_signal = 1.0
                w_bkg    = 1.0
                w_qcd    = 3.0           # increase dominance of QCD

                # Total loss per event
                loss_per_event = (
                    w_signal * base_loss * is_signal
                    + w_bkg    * base_loss * is_bkg
                    + w_qcd    * base_loss * is_qcd
                )

                # ⬅⬅⬅ This is what you asked for:
                batch_loss   = (loss_per_event * w).sum()
                batch_weight = w.sum()

                loss = batch_loss   # or: loss = batch_loss / batch_weight


            # AMP backward
            scaler_amp.scale(loss).backward()

            # Gradient clipping (AMP‑safe)
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler_amp.unscale_(optimizer)
                t.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler_amp.step(optimizer)
            scaler_amp.update()

            # Accumulate epoch totals
            train_loss_sum += batch_loss.item()
            train_weight_sum += batch_weight.item()

        train_loss_optim = train_loss_sum / max(train_weight_sum, 1e-12)
        train_loss, train_acc = evaluate_binary_classifier(
            model,
            train_loader,
            criterion,
            device,
            w_signal=w_signal,
            w_bkg=w_bkg,
            w_qcd=w_qcd,
        )
        val_loss, val_acc = evaluate_binary_classifier(
            model,
            val_loader,
            criterion,
            device,
            w_signal=w_signal,
            w_bkg=w_bkg,
            w_qcd=w_qcd,
        )
        epoch_time = time.time() - epoch_start



        # ------- LR Scheduler & Logging -------
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print_epoch_summary(
            fold_label=fold_label,
            cfg=cfg,
            epoch=epoch,
            n_epochs=cfg.n_epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=current_lr,
            epoch_time=epoch_time,
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_optim": train_loss_optim,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "time_s": epoch_time,
            "qcd_weight_refresh": refresh_qcd,
            "type": "epoch"
        })


        # ----- early stopping -----



        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'variables': training_variables,
            }
        else:
            counter += 1
            if counter >= cfg.patience:
                logger.info("Early stopping for %s at epoch %d", fold_label, epoch)
                break

    # Ensure QCD weights are computed one final time before saving
    model.eval()
    with t.no_grad():

        console.print(
            Rule(
                f"[bold green]Final QCD Refresh[/bold green] | "
                f"{fold_label} | "
            )
        )


        train_pt = refresh_qcd_weights(
            dataset=train_pt,
            model=model,
            qcd_mask_os_loaded=qcd_mask_os_train,
            device=device,
            group_idx=group_idx,
            grouping=grouping,
            use_grouping=use_grouping,
        )

        val_pt = refresh_qcd_weights(
            dataset=val_pt,
            model=model,
            qcd_mask_os_loaded=qcd_mask_os_val,
            device=device,
            group_idx=group_idx,
            grouping=grouping,
            use_grouping=use_grouping,
        )

    if checkpoint is None:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_mean': t.from_numpy(shift.astype(np.float32)),
            'scaler_scale': t.from_numpy(scale.astype(np.float32)),
            'variables': training_variables,
        }


    train_row_index_os = train_pt.row_index.os if hasattr(train_pt, 'row_index') and train_pt.row_index is not None else None
    val_row_index_os   = val_pt.row_index.os   if hasattr(val_pt,   'row_index') and val_pt.row_index   is not None else None

    return model, train_pt.qcd_weights_os, val_pt.qcd_weights_os, train_row_index_os, val_row_index_os



def _predict_ss_weights_for_target_process(
    dataset,
    model,
    *,
    target_process: int,
    device,
    group_idx,
    grouping,
    use_grouping,
):
    output_weights = t.full_like(dataset.weights.ss, fill_value=t.nan)

    model.eval()
    with t.no_grad():
        prediction_ss = predict_probabilities(model, dataset.X.ss, device)

    qcd_mask_ss = dataset.Y.ss == 2
    non_qcd_mask_ss = ~qcd_mask_ss
    target_mask_ss = dataset.process.ss == float(target_process)

    for njets_group in (grouping if use_grouping else ((0, 1000),)):
        if len(njets_group) == 1:
            njets_mask_ss = dataset.X.ss[:, group_idx] == njets_group[0]
        else:
            njets_mask_ss = (
                (dataset.X.ss[:, group_idx] >= njets_group[0])
                & (dataset.X.ss[:, group_idx] <= njets_group[1])
            )

        for sr_value in (True, False):
            sr_mask = dataset.SR_like.ss == sr_value
            qcd_mask = qcd_mask_ss & njets_mask_ss & sr_mask
            non_qcd_mask = non_qcd_mask_ss & njets_mask_ss & sr_mask
            target_mask = target_mask_ss & njets_mask_ss & sr_mask

            if (
                qcd_mask.sum() == 0
                or non_qcd_mask.sum() == 0
                or target_mask.sum() == 0
            ):
                continue

            bins = build_qcd_weight_bins(
                qcd_values=prediction_ss[qcd_mask].squeeze(),
                qcd_weights=dataset.weights.ss[qcd_mask].squeeze(),
                non_qcd_values=prediction_ss[non_qcd_mask].squeeze(),
                non_qcd_weights=dataset.weights.ss[non_qcd_mask].squeeze(),
                binning=QCD_WEIGHT_BINNING,
                n_bins=QCD_WEIGHT_N_BINS,
                dynamic_delta=QCD_SS_WEIGHT_DYNAMIC_DELTA,
                dynamic_delta_last=QCD_SS_WEIGHT_DYNAMIC_DELTA_LAST,
                dynamic_min_qcd_yield=QCD_SS_WEIGHT_DYNAMIC_MIN_QCD_YIELD,
            )
            non_qcd_hist, bins = t.histogram(
                input=prediction_ss[non_qcd_mask],
                bins=bins,
                weight=dataset.weights.ss[non_qcd_mask],
            )
            target_weights = _calculate_scaled_event_weights_generalized(
                prediction_ss[target_mask].squeeze(),
                t.ones_like(prediction_ss[target_mask].squeeze()),
                bins,
                non_qcd_hist,
            )
            output_weights[target_mask] = t.where(
                target_weights < 0,
                t.zeros_like(target_weights),
                target_weights,
            )

    target_unfilled = target_mask_ss & ~t.isfinite(output_weights)
    if target_unfilled.any():
        logger.warning(
            "Setting %d target-process SS QCD extrapolation weights to 0 "
            "because no valid reference bin was available.",
            int(target_unfilled.sum().item()),
        )
        output_weights[target_unfilled] = 0.0

    return output_weights


def train_fold_model_qcd(
        cfg,
        df,
        training_variables,
        group_idx,
        grouping,
        use_grouping,
        device,
        checkpoint_dir,
        fold_label,
        seed: int = TRAINING_SEED,
        qcd_weight_target_process: Union[int, None] = None,
):

    def _split_collection(dataset, ss_idx, os_idx):
        split_fields = {}
        for field in fields(dataset):
            value = getattr(dataset, field.name)
            if value is None:
                split_fields[field.name] = None
            elif hasattr(value, "ss") and hasattr(value, "os"):
                split_fields[field.name] = type(value)(
                    ss=value.ss[ss_idx] if value.ss is not None else None,
                    os=value.os[os_idx] if value.os is not None else None,
                )
            elif field.name.endswith("_os"):
                split_fields[field.name] = value[os_idx]
            else:
                split_fields[field.name] = value
        return type(dataset)(**split_fields)

    train_idx_ss, val_idx_ss = train_test_split(
        np.arange(df.X.ss.shape[0]),
        test_size=0.5,
        random_state=seed,
    )
    train_idx_os, val_idx_os = train_test_split(
        np.arange(df.X.os.shape[0]),
        test_size=0.5,
        random_state=seed,
    )
    train_pt = _split_collection(df, train_idx_ss, train_idx_os).to_torch(device=None)
    val_pt = _split_collection(df, val_idx_ss, val_idx_os).to_torch(device=None)

    X_train = train_pt.X.ss

    shift = X_train.mean(dim=0).to(device)
    scale  = X_train.std(dim=0, unbiased=False).clamp_min(1e-12).to(device)

    model = DNN(
        input_nodes = len(training_variables),
        hidden_nodes = (200, 200),
        output_nodes = 1,
        dropout = 0.15,
        activation = 'ReLU',
        output_activation='Sigmoid',
        input_names = training_variables,
    )

    model.initialize_scaler(shift = shift, scale = scale)

    criterion = nn.BCELoss(reduction='none')                                 
    optimizer = t.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        threshold=cfg.scheduler_threshold,
        threshold_mode='rel',
        cooldown=cfg.scheduler_cooldown,
        min_lr=cfg.scheduler_min_lr,
        eps=cfg.scheduler_eps
    )


    use_amp = (device.type == "cuda") and bool(cfg.use_amp)
    scaler_amp = t.amp.GradScaler('cuda', enabled=use_amp)

    # training loop (full-batch)

    best_val = float('inf')
    counter = 0
    checkpoint = None

    log_rows = []
    logger.info("Starting training for %s", fold_label)

    train_generator = t.Generator().manual_seed(seed)
    val_generator = t.Generator().manual_seed(seed + 1)

    print_training_header(fold_label)

    qcd_mask_ss_train = (train_pt.Y.ss == 2)
    qcd_mask_ss_val = (val_pt.Y.ss == 2)

    for epoch in range(1, cfg.n_epochs + 1):

        refresh_qcd = should_refresh_qcd_weights(epoch)

        # The reference QCD training refreshes the derived weights every epoch.

        if refresh_qcd:
            model.eval()
            with t.no_grad():
                train_pt = refresh_qcd_weights(
                    dataset=train_pt,
                    model=model,
                    qcd_mask_os_loaded=qcd_mask_ss_train,
                    device=device,
                    group_idx=group_idx,
                    grouping=grouping,
                    use_grouping=use_grouping,
                    process="qcd",
                )

                val_pt = refresh_qcd_weights(
                    dataset=val_pt,
                    model=model,
                    qcd_mask_os_loaded=qcd_mask_ss_val,
                    device=device,
                    group_idx=group_idx,
                    grouping=grouping,
                    use_grouping=use_grouping,
                    process="qcd",
                )





        X_train = train_pt.X.ss
        y_train = train_pt.Y.ss
        w_train = train_pt.weights.ss

        X_val = val_pt.X.ss
        y_val = val_pt.Y.ss
        w_val = val_pt.weights.ss


        dataset_train = TensorDataset(X_train, y_train, w_train)
        dataset_val = TensorDataset(X_val, y_val, w_val)


        train_loader = DataLoader(
            dataset_train,
            batch_size = cfg.bsize_train,
            shuffle = True,
            drop_last = False,
            generator=train_generator,
        )

        val_loader = DataLoader(
            dataset_val,
            batch_size = cfg.bsize_val,
            shuffle= True,
            drop_last = False,
            generator=val_generator,
        )

        # ------- train

        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        epoch_start = time.time()

        for Xb, yb, wb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)


            with t.amp.autocast('cuda', enabled=use_amp):

                predictions = model(Xb)
                targets = (yb == 2).float().view(-1, 1)
                weights = wb.float().view(-1, 1)

                loss_per_event = criterion(
                    predictions.float().clamp(1e-7, 1 - 1e-7),
                    targets,
                )
                batch_loss = (loss_per_event * weights).sum()
                batch_weight = weights.sum()
                loss = batch_loss / batch_weight.clamp_min(1e-12)


            # AMP backward
            scaler_amp.scale(loss).backward()

            # Gradient clipping (AMP‑safe)
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler_amp.unscale_(optimizer)
                t.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler_amp.step(optimizer)
            scaler_amp.update()

            # Accumulate epoch totals
            train_loss_sum += batch_loss.item()
            train_weight_sum += batch_weight.item()

        train_loss_optim = train_loss_sum / max(train_weight_sum, 1e-12)
        train_loss, train_acc = _evaluate_qcd_classifier(
            model,
            train_loader,
            criterion,
            device,
        )
        val_loss, val_acc = _evaluate_qcd_classifier(
            model,
            val_loader,
            criterion,
            device,
        )
        epoch_time = time.time() - epoch_start



        # ------- LR Scheduler & Logging -------
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print_epoch_summary(
            fold_label=fold_label,
            cfg=cfg,
            epoch=epoch,
            n_epochs=cfg.n_epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=current_lr,
            epoch_time=epoch_time,
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_optim": train_loss_optim,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "time_s": epoch_time,
            "qcd_weight_refresh": refresh_qcd,
            "type": "epoch"
        })


        # ----- early stopping -----



        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'variables': training_variables,
            }
        else:
            counter += 1
            if counter >= QCD_EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping for %s at epoch %d", fold_label, epoch)
                break

    # Ensure QCD weights are computed one final time before saving
    model.eval()
    with t.no_grad():

        console.print(
            Rule(
                f"[bold green]Final QCD Refresh[/bold green] | "
                f"{fold_label} | "
            )
        )


        train_pt = refresh_qcd_weights(
            dataset=train_pt,
            model=model,
            qcd_mask_os_loaded=qcd_mask_ss_train,
            device=device,
            group_idx=group_idx,
            grouping=grouping,
            use_grouping=use_grouping,
            process="qcd",
        )

        val_pt = refresh_qcd_weights(
            dataset=val_pt,
            model=model,
            qcd_mask_os_loaded=qcd_mask_ss_val,
            device=device,
            group_idx=group_idx,
            grouping=grouping,
            use_grouping=use_grouping,
            process="qcd",
        )

    if checkpoint is None:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_mean': t.from_numpy(shift.astype(np.float32)),
            'scaler_scale': t.from_numpy(scale.astype(np.float32)),
            'variables': training_variables,
        }


    train_row_index_ss = train_pt.row_index.ss if hasattr(train_pt, 'row_index') and train_pt.row_index is not None else None
    val_row_index_ss   = val_pt.row_index.ss   if hasattr(val_pt,   'row_index') and val_pt.row_index   is not None else None
    train_qcd_weights_ss = train_pt.qcd_weights_ss
    val_qcd_weights_ss = val_pt.qcd_weights_ss

    if qcd_weight_target_process is not None:
        train_qcd_weights_ss = _predict_ss_weights_for_target_process(
            train_pt,
            model,
            target_process=qcd_weight_target_process,
            device=device,
            group_idx=group_idx,
            grouping=grouping,
            use_grouping=use_grouping,
        )
        val_qcd_weights_ss = _predict_ss_weights_for_target_process(
            val_pt,
            model,
            target_process=qcd_weight_target_process,
            device=device,
            group_idx=group_idx,
            grouping=grouping,
            use_grouping=use_grouping,
        )

    return model, train_qcd_weights_ss, val_qcd_weights_ss, train_row_index_ss, val_row_index_ss


_train_fold_model_qcd = train_fold_model_qcd


def _train_fold_model_wjets(
        cfg,
        df,
        training_variables,
        group_idx,
        grouping,
        use_grouping,
        device,
        checkpoint_dir,
        fold_label,
    seed: int = TRAINING_SEED,
):
    return _train_fold_model(
        cfg=cfg,
        df=df,
        training_variables=training_variables,
        group_idx=group_idx,
        grouping=grouping,
        use_grouping=use_grouping,
        device=device,
        checkpoint_dir=checkpoint_dir,
        fold_label=fold_label,
    seed=seed,
    )


def _assign_qcd_weights(
    df_target,
    row_indices,
    weights,
    *,
    column_name,
):
    """Assign qcd weights by matching on dataframe row index."""
    if row_indices is None or weights is None:
        return
    target_df = df_target.events if hasattr(df_target, "events") else df_target
    ids_np = row_indices.numpy() if isinstance(row_indices, t.Tensor) else np.asarray(row_indices)
    w_np   = weights.numpy()   if isinstance(weights,   t.Tensor) else np.asarray(weights)
    if ids_np.shape[0] != w_np.shape[0]:
        raise ValueError(
            f"row_indices and weights size mismatch for {column_name}: "
            f"{ids_np.shape[0]} vs {w_np.shape[0]}"
        )
    weight_map = dict(zip(ids_np, w_np))
    mask = target_df.index.isin(weight_map)
    target_df.loc[mask, column_name] = target_df.index.to_series().map(weight_map)


def _select_enrichment_region(df, region_name, additional_masks=()):
    """Return an all-process region from either a configured region or raw masks."""
    manager = df._manager

    if region_name in manager.regions:
        region_mask = manager.get_region_mask(df.events, region_name)
    elif region_name in manager.masks:
        region_mask = manager.get_mask(df.events, region_name)
    else:
        raise ValueError(f"Unknown enrichment region or mask: {region_name}")

    for mask_name in additional_masks:
        if mask_name not in manager.masks:
            raise ValueError(f"Unknown enrichment mask: {mask_name}")
        region_mask &= manager.get_mask(df.events, mask_name)

    return df.full.events.loc[region_mask]


def _select_enrichment_regions(df, region_names, additional_masks=()):
    """Return a union of configured all-process regions."""
    frames = [
        _select_enrichment_region(
            df,
            region_name,
            additional_masks=additional_masks,
        )
        for region_name in region_names
    ]
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, axis=0).loc[
        lambda frame: ~frame.index.duplicated(keep="first")
    ].copy()


def _drop_processes_from_region(region_df, df, excluded_processes=()):
    """Drop configured processes from an already selected region dataframe."""
    excluded_processes = tuple(excluded_processes)
    if not excluded_processes:
        return region_df

    drop_mask = pd.Series(False, index=region_df.index)
    for process_name in excluded_processes:
        if process_name not in df._manager.processes:
            raise ValueError(f"Unknown process to exclude: {process_name}")
        drop_mask |= df._manager.get_process_mask(region_df, process_name)

    logger.info(
        "Excluding %d rows from processes %s in enrichment region.",
        int(drop_mask.sum()),
        ", ".join(excluded_processes),
    )
    return region_df.loc[~drop_mask].copy()


def _validate_qcd_fraction_sign_regions(df, region_df):
    """Ensure the fractions training region differs from AR only by charge sign."""
    selected_ss = region_df.index[region_df["SS"]]
    selected_os = region_df.index[region_df["OS"]]
    expected_ss = df.full.AR_SS.events.index
    expected_os = df.full.AR.events.index

    if not selected_ss.equals(expected_ss):
        missing = len(expected_ss.difference(selected_ss))
        extra = len(selected_ss.difference(expected_ss))
        raise ValueError(
            "DR_qcd_fractions_no_signs SS selection does not match AR_SS: "
            f"missing={missing}, extra={extra}"
        )

    if not selected_os.equals(expected_os):
        missing = len(expected_os.difference(selected_os))
        extra = len(selected_os.difference(expected_os))
        raise ValueError(
            "DR_qcd_fractions_no_signs OS selection does not match AR: "
            f"missing={missing}, extra={extra}"
        )

    logger.info(
        "QCD fractions sign-region validation passed: SS=%d, OS=%d",
        len(selected_ss),
        len(selected_os),
    )


# -------------- tasks ----------------

def _run_enrichment_process(
    process_name: str,
    input_file_path: Path,
    train_fold_model_fn,
    region_name: str,
    output_root: Path = None,
    output_name: str = None,
    feature_column_name: str = None,
    feature_file_prefix: str = None,
    additional_region_masks: Tuple[str, ...] = (),
    extra_region_names: Tuple[str, ...] = (),
    validate_qcd_fraction_sign_regions: bool = False,
    excluded_processes: Tuple[str, ...] = (),
    data_loader=load_data,
    qcd_weight_target_process: Union[int, None] = None,
):
    _set_training_seed(TRAINING_SEED)

    project_root = PROJECT_ROOT
    output_root = Path(output_root) if output_root is not None else project_root
    process_slug = (output_name or process_name).lower()
    DATA_PATH = Path(input_file_path)
    MASKS_PATH = CONFIG_ROOT / 'selections.yaml'
    TRAINING_VARIABLES_ENRICHMENT = CONFIG_ROOT / 'variables_enrichment.yaml'
    CONFIG_MODEL_PATH = CONFIG_ROOT / 'model_enrichment.yaml'
    CHECKPOINT_DIR = output_root / 'Enrichment_models' / process_slug
    FEATURE_REGISTRY_PATH = output_root / 'data' / 'features' / process_slug / 'feature_registry.json'
    FEATURE_STORE_DIR = output_root / 'data' / 'features' / process_slug
    DEFAULT_FEATURE_REGISTRY_PATH = output_root / 'data' / 'features' / 'feature_registry.json'


    device = t.device("cuda" if t.cuda.is_available() else "cpu")


    df = data_loader(DATA_PATH, MASKS_PATH)
    training_variables = load_variables(TRAINING_VARIABLES_ENRICHMENT)

    raw = load_config(CONFIG_MODEL_PATH)
    config = Config.from_dict(raw)
    grouping_cfg = _load_qcd_grouping_config(QCD_GROUPING_CONFIG_PATH)
    _validate_grouping_config(training_variables, grouping_cfg)
    logger.info(
        "Loaded enrichment groupings from %s: %s",
        Path(QCD_GROUPING_CONFIG_PATH).resolve(),
        {
            item["name"]: item["groups"]
            for item in grouping_cfg["groupings"]
        },
    )



    region_names = (region_name, *tuple(extra_region_names))
    region_df = _select_enrichment_regions(
        df,
        region_names,
        additional_masks=additional_region_masks,
    )
    region_df = _drop_processes_from_region(
        region_df,
        df,
        excluded_processes=excluded_processes,
    )
    if validate_qcd_fraction_sign_regions:
        _validate_qcd_fraction_sign_regions(df, region_df)
    data_pt_even = get_my_data(region_df[region_df.event % 2 == 0], training_variables)
    data_pt_odd = get_my_data(region_df[region_df.event % 2 == 1], training_variables)

    base_path = Path(CHECKPOINT_DIR)
    feature_files = []

    for item in grouping_cfg["groupings"]:
        grouping_name = item["name"]
        group_idx_name = item["index_name"]
        grouping = item["groups"]
        group_idx = training_variables.index(group_idx_name)
        current_feature_column = (
            f"{feature_column_name}_{grouping_name}"
            if feature_column_name is not None
            else f"qcd_weight_{grouping_name}"
        )
        region_df[current_feature_column] = np.nan
        logger.info(
            "Starting enrichment training with grouping '%s' from %s "
            "at index %d and bins=%s",
            grouping_name,
            group_idx_name,
            group_idx,
            grouping,
        )
        target_weight_kwargs = {}
        if qcd_weight_target_process is not None:
            target_weight_kwargs["qcd_weight_target_process"] = (
                qcd_weight_target_process
            )

        even_model, qcd_weights_train_even, qcd_weights_val_even, train_rows_even, val_rows_even = train_fold_model_fn(
            cfg=config,
            training_variables=training_variables,
            group_idx=group_idx,
            grouping=grouping,
            use_grouping=grouping_cfg["use_grouping"],
            df=data_pt_odd,
            device=device,
            checkpoint_dir=CHECKPOINT_DIR,
            fold_label=f'fold_even_{grouping_name}',
            seed=TRAINING_SEED,
            **target_weight_kwargs,
        )

        odd_model, qcd_weights_train_odd, qcd_weights_val_odd, train_rows_odd, val_rows_odd = train_fold_model_fn(
            cfg=config,
            training_variables=training_variables,
            group_idx=group_idx,
            grouping=grouping,
            use_grouping=grouping_cfg["use_grouping"],
            df=data_pt_even,
            device=device,
            checkpoint_dir=CHECKPOINT_DIR,
            fold_label=f'fold_odd_{grouping_name}',
            seed=TRAINING_SEED,
            **target_weight_kwargs,
        )

        model = FoldCombinedDNN(
            even_model=even_model,
            odd_model=odd_model,
            fold_id_name='parity',
        )

        group_base_path = base_path / grouping_name
        save_model(even_model, group_base_path / 'fold_even')
        save_model(odd_model, group_base_path / 'fold_odd')
        save_model(model, group_base_path / 'combined')

        _assign_qcd_weights(
            region_df,
            train_rows_even,
            qcd_weights_train_even,
            column_name=current_feature_column,
        )
        _assign_qcd_weights(
            region_df,
            val_rows_even,
            qcd_weights_val_even,
            column_name=current_feature_column,
        )
        _assign_qcd_weights(
            region_df,
            train_rows_odd,
            qcd_weights_train_odd,
            column_name=current_feature_column,
        )
        _assign_qcd_weights(
            region_df,
            val_rows_odd,
            qcd_weights_val_odd,
            column_name=current_feature_column,
        )

        current_feature_prefix = feature_file_prefix or "qcd_weights"
        feature_store_path = (
            FEATURE_STORE_DIR
            / f'{current_feature_prefix}_{grouping_name}.feather'
        )

        registry = FeatureRegistry(FEATURE_REGISTRY_PATH)
        store = FeatureStore(feature_store_path, registry)

        store.write(pd.DataFrame({
            "row_index": region_df.index,
            "event": region_df["event"],
            current_feature_column: region_df[current_feature_column],
        }))

        store.save()
        registry.save()

        default_registry = FeatureRegistry(DEFAULT_FEATURE_REGISTRY_PATH)
        default_registry.register(
            [current_feature_column],
            feature_store_path,
        )
        default_registry.save()

        feature_files.append(feature_store_path)

        _write_group_metadata(
            group_base_path / "metadata.json",
            {
                "grouping_name": grouping_name,
                "grouping_variable": group_idx_name,
                "group_index": group_idx,
                "groups": [list(g) for g in grouping],
                "use_grouping": grouping_cfg["use_grouping"],
                "default_grouping": {
                    "index_name": grouping_cfg["default_grouping"]["index_name"],
                    "groups": [list(g) for g in grouping_cfg["default_grouping"]["groups"]],
                },
                "training_variables_file": str(TRAINING_VARIABLES_ENRICHMENT),
                "model_config_file": str(CONFIG_MODEL_PATH),
                "region_name": region_name,
                "additional_region_masks": list(additional_region_masks),
                "excluded_processes": list(excluded_processes),
                "feature_column": current_feature_column,
                "feature_file": str(feature_store_path),
            },
        )

        logger.info("Finished grouping '%s'. Model artifacts at %s", grouping_name, group_base_path)

    return {
        "process_name": process_slug,
        "combined_model_path": base_path,
        "feature_file": FEATURE_STORE_DIR,
        "feature_files": feature_files,
    }


def train_enrichment_wjets(input_file_path: Path, output_root: Path = None):
    return _run_enrichment_process(
        process_name="wjets",
        input_file_path=input_file_path,
        train_fold_model_fn=_train_fold_model_wjets,
        region_name="DR_wjets_without_signs",
        output_root=output_root,
    )

def train_enrichment_qcd(input_file_path: Path, output_root: Path = None):
    return _run_enrichment_process(
        process_name="qcd",
        input_file_path=input_file_path,
        train_fold_model_fn=train_fold_model_qcd,
        region_name="DR_qcd_without_signs",
        output_root=output_root,
    )


def train_enrichment_qcd_fractions(
    input_file_path: Path,
    output_root: Path = None,
):
    """Train QCD enrichment in the fractions region for every grouping."""
    return _run_enrichment_process(
        process_name="qcd",
        input_file_path=input_file_path,
        train_fold_model_fn=train_fold_model_qcd,
        region_name="DR_qcd_fractions_no_signs",
        output_name="qcd_fraction",
        feature_column_name="weight_qcd_fraction",
        feature_file_prefix="qcd_fraction_weights",
        validate_qcd_fraction_sign_regions=True,
        output_root=output_root,
    )


def train_enrichment_qcd_extrapolation(
    input_file_path: Path,
    output_root: Path = None,
):
    """Train QCD extrapolation weights for same-sign AR and SR regions."""
    return _run_enrichment_process(
        process_name="qcd",
        input_file_path=input_file_path,
        train_fold_model_fn=train_fold_model_qcd,
        region_name="AR_SS",
        extra_region_names=("SR_SS",),
        output_name="qcd_extrapolation",
        feature_column_name="weight_qcd_extrapolation",
        feature_file_prefix="qcd_extrapolation_weights",
        excluded_processes=("embedding",),
        data_loader=load_data_no_embedding,
        qcd_weight_target_process=0,
        output_root=output_root,
    )


if __name__ == "__main__":
    train_enrichment_wjets('../data/dataframe_complete.feather')
