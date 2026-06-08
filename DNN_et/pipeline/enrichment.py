import numpy as np
from pathlib import Path
import logging
import json
import random
import torch as t
import torch.nn as nn
import pandas as pd
from typing import Union, Any, Dict, Tuple, Literal
from classes import load_data, load_variables, load_config, FeatureStore, FeatureRegistry, save_model
from classes.helper import get_class_weights, _same_sign_opposite_sign_split, _collection, _component_collection
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, fields
from classes.CustomLogging import setup_logging
from classes import DNN, FoldCombinedDNN
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from classes.enrichment_classifier import get_my_data, should_refresh_qcd_weights, refresh_qcd_weights, evaluate_binary_classifier
from rich.console import Console
from rich.table import Table
from rich.rule import Rule

import time

logger = setup_logging(logger=logging.getLogger(__name__))
console = Console()

# ---------- dataclasses -------------


QCD_WEIGHT_BINNING = 'dynamic'
QCD_WEIGHT_N_BINS = 20
QCD_WEIGHT_DYNAMIC_DELTA = 10.0
QCD_WEIGHT_DYNAMIC_DELTA_LAST = 10.0
QCD_WEIGHT_DYNAMIC_MIN_QCD_YIELD = 10.0
QCD_WEIGHT_REFRESH_EVERY = 5
QCD_WEIGHT_REFRESH_UNTIL_EPOCH = 100
QCD_GROUPING_CONFIG_PATH = "../configs/config_qcd_groupings_enrichment.yaml"
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
                "index_name": item["index_name"],
                "groups": _normalize_groups(item["groups"]),
            }
            for item in qcd_cfg["groupings"]
        ],
    }


def _validate_grouping_config(training_variables, grouping_cfg):
    known_vars = set(training_variables)
    for item in grouping_cfg["groupings"]:
        name = item["index_name"]
        groups = item["groups"]
        if name not in known_vars:
            raise ValueError(f"Grouping variable '{name}' not found in training variables")
        if not isinstance(groups, tuple) or len(groups) == 0:
            raise ValueError(f"Grouping for '{name}' must be a non-empty tuple of groups")
        for group in groups:
            if len(group) not in (1, 2):
                raise ValueError(
                    f"Invalid group {group} for '{name}'. Each group must be (value,) or (min,max)."
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

    return model, train_pt.qcd_weights_ss, val_pt.qcd_weights_ss, train_row_index_ss, val_row_index_ss


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


def _assign_qcd_weights(df_target, row_indices, weights, suffix):
    """Assign qcd weights by matching on dataframe row index."""
    if row_indices is None or weights is None:
        return
    target_df = df_target.events if hasattr(df_target, "events") else df_target
    ids_np = row_indices.numpy() if isinstance(row_indices, t.Tensor) else np.asarray(row_indices)
    w_np   = weights.numpy()   if isinstance(weights,   t.Tensor) else np.asarray(weights)
    if ids_np.shape[0] != w_np.shape[0]:
        raise ValueError(
            f"row_indices and weights size mismatch for {suffix}: "
            f"{ids_np.shape[0]} vs {w_np.shape[0]}"
        )
    weight_map = dict(zip(ids_np, w_np))
    mask = target_df.index.isin(weight_map)
    target_df.loc[mask, f'qcd_weight_{suffix}'] = target_df.index.to_series().map(weight_map)
# -------------- tasks ----------------

def _run_enrichment_process(
    process_name: str,
    input_file_path: Path,
    train_fold_model_fn,
    region_name: str,
):
    _set_training_seed(TRAINING_SEED)

    project_root = Path(__file__).resolve().parent.parent
    process_slug = process_name.lower()
    DATA_PATH = Path(input_file_path)
    MASKS_PATH = project_root / 'configs' / 'masks.yaml'
    TRAINING_VARIABLES_ENRICHMENT = project_root / 'configs' / 'training_variables_enrichment.yaml'
    CONFIG_MODEL_PATH = project_root / 'configs' / 'config_NN_enrichment.yaml'
    CHECKPOINT_DIR = project_root / 'Enrichment_models' / process_slug
    FEATURE_REGISTRY_PATH = project_root / 'data' / 'features' / process_slug / 'feature_registry.json'
    FEATURE_STORE_DIR = project_root / 'data' / 'features' / process_slug
    DEFAULT_FEATURE_REGISTRY_PATH = project_root / 'data' / 'features' / 'feature_registry.json'


    device = t.device("cuda" if t.cuda.is_available() else "cpu")


    df = load_data(DATA_PATH, MASKS_PATH)
    training_variables = load_variables(TRAINING_VARIABLES_ENRICHMENT)

    raw = load_config(CONFIG_MODEL_PATH)
    config = Config.from_dict(raw)
    grouping_cfg = _load_qcd_grouping_config(QCD_GROUPING_CONFIG_PATH)
    _validate_grouping_config(training_variables, grouping_cfg)



    region_df = getattr(df.data, region_name).events
    data_pt_even = get_my_data(region_df[region_df.event % 2 == 0], training_variables)
    data_pt_odd = get_my_data(region_df[region_df.event % 2 == 1], training_variables)

    base_path = Path(CHECKPOINT_DIR)
    feature_files = []

    for item in grouping_cfg["groupings"]:
        group_idx_name = item["index_name"]
        grouping = item["groups"]
        group_idx = training_variables.index(group_idx_name)
        logger.info(
            "Starting enrichment training with grouping '%s' at index %d and bins=%s",
            group_idx_name,
            group_idx,
            grouping,
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
            fold_label=f'fold_even_{group_idx_name}',
            seed=TRAINING_SEED,
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
            fold_label=f'fold_odd_{group_idx_name}',
            seed=TRAINING_SEED,
        )

        model = FoldCombinedDNN(
            even_model=even_model,
            odd_model=odd_model,
            fold_id_name='parity',
        )

        group_base_path = base_path / group_idx_name
        save_model(even_model, group_base_path / 'fold_even')
        save_model(odd_model, group_base_path / 'fold_odd')
        save_model(model, group_base_path / 'combined')

        _assign_qcd_weights(region_df, train_rows_even, qcd_weights_train_even, group_idx_name)
        _assign_qcd_weights(region_df, val_rows_even,   qcd_weights_val_even, group_idx_name)
        _assign_qcd_weights(region_df, train_rows_odd, qcd_weights_train_odd, group_idx_name)
        _assign_qcd_weights(region_df, val_rows_odd,   qcd_weights_val_odd, group_idx_name)
        
        feature_store_path = FEATURE_STORE_DIR / f'qcd_weights_{group_idx_name}.feather'

        registry = FeatureRegistry(FEATURE_REGISTRY_PATH)
        store = FeatureStore(feature_store_path, registry)

        store.write(pd.DataFrame({
            "row_index": region_df.index,
            "event": region_df["event"],
            f"qcd_weight_{group_idx_name}": region_df[f"qcd_weight_{group_idx_name}"],
        }))

        store.save()
        registry.save()

        default_registry = FeatureRegistry(DEFAULT_FEATURE_REGISTRY_PATH)
        default_registry.register([f"qcd_weight_{group_idx_name}"], feature_store_path)
        default_registry.save()

        feature_files.append(feature_store_path)

        _write_group_metadata(
            group_base_path / "metadata.json",
            {
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
            },
        )

        logger.info("Finished grouping '%s'. Model artifacts at %s", group_idx_name, group_base_path)

    return {
        "process_name": process_slug,
        "combined_model_path": base_path,
        "feature_file": FEATURE_STORE_DIR,
        "feature_files": feature_files,
    }


def train_enrichment_wjets(input_file_path: Path):
    return _run_enrichment_process(
        process_name="wjets",
        input_file_path=input_file_path,
        train_fold_model_fn=_train_fold_model_wjets,
        region_name="DR_wjets_without_signs",
    )

def train_enrichment_qcd(input_file_path: Path):
    return _run_enrichment_process(
        process_name="qcd",
        input_file_path=input_file_path,
        train_fold_model_fn=train_fold_model_qcd,
        region_name="DR_qcd_without_signs",
    )

if __name__ == "__main__":
    train_enrichment_wjets('../data/dataframe_complete.feather')
