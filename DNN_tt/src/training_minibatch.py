import torch as t
import numpy as np
import random
import logging
import yaml
from classes import load_variables, load_data, create_training_dataset
from classes import DNN, GroupedDNN, FoldCombinedDNN
from classes import save_model
from dataclasses import dataclass, MISSING
from typing import List, Optional, Union, Tuple, Dict, Any
import yaml
from dataclasses import is_dataclass, fields
from classes.Logging import Logging
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import copy


SEED = 42
logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()

    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        root.setLevel(level)

    logger.setLevel(level)


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

DATA_PATH = '../data/data_complete.feather'
MASKS_PATH = 'configs/masks.yaml'
TRAINING_VAR_PATH = 'configs/training_variables.yaml'
NN_CONFIG_PATH = 'configs/DNN.yaml'
CHECKPOINT_DIR = 'Training_results'



def load_variables(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    yaml_vars = config.get("variables", [])
    return yaml_vars

def load_config(path: str, cls):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return _from_dict(data, cls)

def _from_dict(data: dict, cls):
    """
    Minimal recursive dict → dataclass converter
    """

    if not is_dataclass(cls):
        return data

    kwargs = {}

    for field in fields(cls):
        has_value = field.name in data
        value = data.get(field.name) if has_value else MISSING

        if value is MISSING:
            if field.default is not MISSING:
                kwargs[field.name] = field.default
            elif field.default_factory is not MISSING:
                kwargs[field.name] = field.default_factory()
            else:
                kwargs[field.name] = None
            continue

        # tuple conversion (important for hidden_nodes)
        if field.type == tuple or field.type == Tuple[int, ...]:
            kwargs[field.name] = tuple(value)

        # nested dataclass
        elif is_dataclass(field.type):
            kwargs[field.name] = _from_dict(value, field.type)

        else:
            kwargs[field.name] = value

    return cls(**kwargs)

@dataclass
class ModelConfig:
    hidden_nodes: Tuple[int, ...]
    output_nodes: int

    activation: str = "ReLU"
    output_activation: str = "Sigmoid"

    dropout: Union[float, Tuple[float, ...]] = 0.1

@dataclass
class TrainingConfig:
    epochs: int = 50
    lr: float = 1e-3
    loss: str = "BCE"
    batch_size: int = 10000
    val_every_epochs: int = 5
    log_batch_progress: bool = False
    log_batch_parts: int = 5
    fullbatch_validation: bool = True
    use_amp: bool = True

@dataclass
class SchedulerConfig:
    patience: int = 10
    early_stopping_patience: int = 20
    factor: float = 0.1
    min_delta: float = 1.0e-4
    min_lr: float = 1e-6

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    scheduler: SchedulerConfig


def _run_epoch_minibatch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    training: bool,
    epoch: int,
    epochs: int,
    log_batch_progress: bool,
    log_batch_parts: int,
    use_amp: bool,
    scaler,
):
    if training:
        model.train()
    else:
        model.eval()

    total_weighted_loss = 0.0
    total_weight = 0.0

    num_batches = len(loader)
    log_every = max(1, num_batches // max(1, log_batch_parts))
    phase = "train" if training else "val"
    amp_enabled = use_amp and (str(device).startswith("cuda"))

    with t.set_grad_enabled(training):
        for batch_idx, (X_batch, Y_batch, W_batch) in enumerate(loader, start=1):
            X_batch = X_batch.to(device)

            expected_features = int(getattr(model, "_input_nodes", 1))

            # Ensure model input is [batch, expected_features].
            if X_batch.dim() == 1:
                if expected_features == 1:
                    X_batch = X_batch.unsqueeze(-1)
                elif X_batch.numel() % expected_features == 0:
                    X_batch = X_batch.reshape(-1, expected_features)
                else:
                    raise ValueError(
                        f"Cannot reshape 1D minibatch of length {X_batch.numel()} "
                        f"to expected feature size {expected_features}."
                    )
            elif X_batch.dim() == 2 and X_batch.shape[1] != expected_features:
                if X_batch.numel() % expected_features == 0:
                    X_batch = X_batch.reshape(-1, expected_features)
                else:
                    raise ValueError(
                        f"Minibatch feature mismatch: got {tuple(X_batch.shape)}, "
                        f"expected second dim {expected_features}."
                    )
            elif X_batch.dim() > 2:
                if X_batch.numel() % expected_features == 0:
                    X_batch = X_batch.reshape(-1, expected_features)
                else:
                    raise ValueError(
                        f"Unsupported minibatch rank {X_batch.dim()} with shape {tuple(X_batch.shape)}."
                    )

            # Keep target/weights 1D even for single-item batches.
            Y_batch = Y_batch.to(device).reshape(-1)
            W_batch = W_batch.to(device).reshape(-1)

            with t.amp.autocast(device_type="cuda", enabled=amp_enabled):
                # Ensure prediction shape matches BCE target shape.
                preds = model(X_batch).reshape(-1)

            # BCE/BCELoss is not autocast-safe; compute loss in float32.
            preds_fp32 = preds.float()
            y_fp32 = Y_batch.float()
            w_fp32 = W_batch.float()

            per_sample_loss = loss_fn(preds_fp32, y_fp32)
            weight_sum = w_fp32.sum()
            batch_loss = (per_sample_loss * w_fp32).sum() / (weight_sum + 1e-12)

            if training:
                optimizer.zero_grad()
                if amp_enabled:
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    batch_loss.backward()
                    optimizer.step()

            total_weighted_loss += (per_sample_loss * w_fp32).sum().item()
            total_weight += weight_sum.item()

            if log_batch_progress and (batch_idx == 1 or batch_idx == num_batches or batch_idx % log_every == 0):
                running_loss = total_weighted_loss / (total_weight + 1e-12)
                logger.info(
                    "Epoch %d/%d [%s] batch %d/%d (%.0f%%) running_loss=%.6f",
                    epoch,
                    epochs,
                    phase,
                    batch_idx,
                    num_batches,
                    100.0 * batch_idx / num_batches,
                    running_loss,
                )

    return total_weighted_loss / (total_weight + 1e-12)


def _run_validation_fullbatch(model, val, loss_fn, device, use_amp: bool):
    model.eval()

    X = val.X.to(device)
    Y = val.Y.to(device).reshape(-1)
    W = val.weights.to(device).reshape(-1)

    expected_features = int(getattr(model, "_input_nodes", 1))
    if X.dim() == 1:
        if expected_features == 1:
            X = X.unsqueeze(-1)
        elif X.numel() % expected_features == 0:
            X = X.reshape(-1, expected_features)
        else:
            raise ValueError(
                f"Cannot reshape 1D validation tensor of length {X.numel()} to expected feature size {expected_features}."
            )
    elif X.dim() == 2 and X.shape[1] != expected_features:
        if X.numel() % expected_features == 0:
            X = X.reshape(-1, expected_features)
        else:
            raise ValueError(
                f"Validation feature mismatch: got {tuple(X.shape)}, expected second dim {expected_features}."
            )

    amp_enabled = use_amp and (str(device).startswith("cuda"))
    with t.no_grad():
        with t.amp.autocast(device_type="cuda", enabled=amp_enabled):
            preds = model(X).reshape(-1)

        preds_fp32 = preds.float()
        y_fp32 = Y.float()
        w_fp32 = W.float()
        per_sample_loss = loss_fn(preds_fp32, y_fp32)
        val_loss = (per_sample_loss * w_fp32).sum() / (w_fp32.sum() + 1e-12)

    return val_loss.item()


def train_dnn_minibatch(
    model,
    train,
    val,
    epochs: int,
    batch_size: int,
    lr: float,
    device,
    scheduler_patience: int,
    early_stopping_patience: int,
    scheduler_factor: float,
    min_delta: float,
    min_lr: float,
    val_every_epochs: int,
    log_batch_progress: bool,
    log_batch_parts: int,
    fullbatch_validation: bool,
    use_amp: bool,
):
    model = model.to(device)

    train_ds = TensorDataset(train.X, train.Y, train.weights)
    val_ds = TensorDataset(val.X, val.Y, val.weights)

    pin_memory = str(device).startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory)

    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )
    loss_fn = t.nn.BCELoss(reduction="none")

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    scaler = t.amp.GradScaler(enabled=(use_amp and str(device).startswith("cuda")))

    logger.info(
        "Start training: epochs=%d, batch_size=%d, train_samples=%d, val_samples=%d, device=%s",
        epochs,
        batch_size,
        len(train_ds),
        len(val_ds),
        device,
    )
    logger.info(
        "Fast mode: val_every_epochs=%d, fullbatch_validation=%s, log_batch_progress=%s, use_amp=%s",
        val_every_epochs,
        fullbatch_validation,
        log_batch_progress,
        use_amp,
    )

    for epoch in range(epochs):
        train_loss = _run_epoch_minibatch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            training=True,
            epoch=epoch + 1,
            epochs=epochs,
            log_batch_progress=log_batch_progress,
            log_batch_parts=log_batch_parts,
            use_amp=use_amp,
            scaler=scaler,
        )

        run_validation = ((epoch + 1) % max(1, val_every_epochs) == 0) or (epoch + 1 == epochs)
        if run_validation:
            if fullbatch_validation:
                val_loss = _run_validation_fullbatch(
                    model=model,
                    val=val,
                    loss_fn=loss_fn,
                    device=device,
                    use_amp=use_amp,
                )
            else:
                val_loss = _run_epoch_minibatch(
                    model=model,
                    loader=val_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device,
                    training=False,
                    epoch=epoch + 1,
                    epochs=epochs,
                    log_batch_progress=log_batch_progress,
                    log_batch_parts=log_batch_parts,
                    use_amp=use_amp,
                    scaler=scaler,
                )
        else:
            val_loss = None

        old_lr = optimizer.param_groups[0]["lr"]
        if val_loss is not None:
            scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr != old_lr:
            logger.info("Learning rate reduced: %.2e -> %.2e", old_lr, new_lr)

        if val_loss is not None:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            logger.info(
                "Epoch %d/%d - train_loss=%.6f val_loss=%.6f lr=%.2e patience=%d",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                optimizer.param_groups[0]["lr"],
                epochs_without_improvement,
            )
        else:
            logger.info(
                "Epoch %d/%d - train_loss=%.6f val=skipped lr=%.2e",
                epoch + 1,
                epochs,
                train_loss,
                optimizer.param_groups[0]["lr"],
            )

        if val_loss is not None and epochs_without_improvement >= early_stopping_patience:
            logger.info("Early stopping triggered after %d epochs", epoch + 1)
            break

        if optimizer.param_groups[0]["lr"] <= min_lr:
            logger.info("Stopping: minimum learning rate reached")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info("Best validation loss: %.6f", best_val_loss)
    return model, best_val_loss


def _train_fold_model(cfg, grouping, training_var, df_sig, df_bkg, weight_column, device):
    train, val = create_training_dataset(
        df_sig=df_sig,
        df_bkg=df_bkg,
        training_var=training_var,
        weight_column=weight_column,
        balance=True,
        test_size=0.25,
        random_state=SEED,
    )

    base_model = DNN(
        input_nodes=train.X.shape[1],
        hidden_nodes=cfg.model.hidden_nodes,
        output_nodes=1,
        activation=cfg.model.activation,
        output_activation=cfg.model.output_activation,
        dropout=cfg.model.dropout,
        input_names=training_var,
    )

    model = GroupedDNN(
        grouping=grouping,
        default_model=base_model,
    )

    model.initialize_scaler(
        shift=train.X.mean(dim=0),
        scale=train.X.std(dim=0) + 1e-6,
    )

    model, best_loss = train_dnn_minibatch(
        model=model,
        train=train,
        val=val,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        device=device,
        scheduler_patience=cfg.scheduler.patience,
        early_stopping_patience=cfg.scheduler.early_stopping_patience,
        scheduler_factor=cfg.scheduler.factor,
        min_delta=cfg.scheduler.min_delta,
        min_lr=cfg.scheduler.min_lr,
        val_every_epochs=cfg.training.val_every_epochs,
        log_batch_progress=cfg.training.log_batch_progress,
        log_batch_parts=cfg.training.log_batch_parts,
        fullbatch_validation=cfg.training.fullbatch_validation,
        use_amp=cfg.training.use_amp,
    )

    return model


def main():

    setup_logging(logging.INFO)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    cfg = load_config(NN_CONFIG_PATH, Config)

    df = load_data(DATA_PATH, MASKS_PATH)

    training_var = load_variables(TRAINING_VAR_PATH)

    taudm_idx = training_var.index('tau_decaymode_2')
    njets_idx = training_var.index('njets')

    grouping_taudm = {
        taudm_idx: (
            (0,),
            (1,),
            (10,),
            (11,),
        )
    }

    grouping_njets = {
        njets_idx: (
            (0,),
            (1,),
            (2, 1000),
        )
    }

    for grouping, group_label in zip([grouping_taudm, grouping_njets], ['tau_decaymode', 'njets']):

        logger.info('Group splitting: %s', group_label)

        for process in ['wjets', 'qcd']:

            if process == 'wjets':
                df_sig = df.data.SR_like_wjets
                df_bkg = df.data.AR_like_wjets
                weight_column = 'weight_wjets'
            elif process == 'qcd':
                df_sig = df.data.SR_like_qcd
                df_bkg = df.data.AR_like_qcd
                weight_column = 'weight_qcd'

            df_sig_plain = df_sig.events
            df_bkg_plain = df_bkg.events
            df_sig_even = df_sig_plain[df_sig_plain['event']%2 == 0]
            df_sig_odd  = df_sig_plain[df_sig_plain['event']%2 == 1]
            df_bkg_even = df_bkg_plain[df_bkg_plain['event']%2 == 0]
            df_bkg_odd  = df_bkg_plain[df_bkg_plain['event']%2 == 1]

            logger.info(
                "%s/%s fold sizes: even=%d (sig=%d, bkg=%d), odd=%d (sig=%d, bkg=%d)",
                group_label,
                process,
                len(df_sig_even) + len(df_bkg_even),
                len(df_sig_even),
                len(df_bkg_even),
                len(df_sig_odd) + len(df_bkg_odd),
                len(df_sig_odd),
                len(df_bkg_odd),
            )

            # even_model: trained on odd events, applied to even events
            logger.info("Start fold training: %s/%s -> model for even events (train on odd)", group_label, process)
            even_model = _train_fold_model(
                cfg=cfg,
                grouping=grouping,
                training_var=training_var,
                df_sig=df_sig_odd,
                df_bkg=df_bkg_odd,
                weight_column=weight_column,
                device=device,
            )

            # odd_model: trained on even events, applied to odd events
            logger.info("Start fold training: %s/%s -> model for odd events (train on even)", group_label, process)
            odd_model = _train_fold_model(
                cfg=cfg,
                grouping=grouping,
                training_var=training_var,
                df_sig=df_sig_even,
                df_bkg=df_bkg_even,
                weight_column=weight_column,
                device=device,
            )

            model = FoldCombinedDNN(
                even_model=even_model,
                odd_model=odd_model,
                fold_id_name='event',
            )

            base_path = Path(CHECKPOINT_DIR) / group_label / process
            save_model(even_model, base_path / 'fold_even')
            save_model(odd_model, base_path / 'fold_odd')
            save_model(model, base_path)


if __name__ == '__main__':
    main()