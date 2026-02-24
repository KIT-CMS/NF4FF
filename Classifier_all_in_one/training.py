import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from CustomLogging import setup_logging
#from CustomLogging import setup_logging
from classes.path_managment import StorePathHelper
from classes.config_loader import load_config
import CODE.HELPER as helper
import torch as t
from tap import Tap
from typing import Any, Callable, Dict, Generator, List, Literal, Tuple, Union
from copy import deepcopy

import time


# ----- seed -----

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ------ logger -----

logger = setup_logging(logger=logging.getLogger(__name__))

# ----- TAP Arguments -----

# ----- constants ------

INPUT_DIM = 20
CONFIG_MODEL_PATH = 'configs/config_NN.yaml'
PATIENCE = 80

# ----- data classes -----
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
    
@dataclass
class _same_sign_opposite_sign_split(metaclass=helper.CollectionMeta):
    ss: Union[t.Tensor, pd.DataFrame, np.ndarray]
    os: Union[t.Tensor, pd.DataFrame, np.ndarray]

@dataclass
class _component_collection(metaclass=helper.CollectionMeta):
    X: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    Y: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    weights: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    class_weights: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None

@dataclass
class _collection:
    values: Any
    weights: Any
    histograms: Any
    
    @property
    def unrolled(self) -> tuple[Any, ...]:
        return (self.values, self.weights, self.histograms)


# ------ lists -----
dim = 20
variables = [
    "pt_1","pt_2","eta_1","eta_2","jpt_1","jpt_2","jeta_1","jeta_2",
    "m_fastmtt","pt_fastmtt","met","njets","nbtag","mt_tot","m_vis",
    "pt_tt","pt_vis","mjj","pt_dijet","pt_ttjj","pzetamissvis","deltaEta_jj",
    "deltaEta_ditaupair","deltaR_jj","deltaR_ditaupair","deltaR_1j1","deltaR_1j2",
    "deltaR_2j1","deltaR_2j2","deltaR_12j1","deltaR_12j2","deltaEta_1j1",
    "deltaEta_1j2","deltaEta_2j1","deltaEta_2j2","deltaEta_12j1","deltaEta_12j2",
][:dim]

# ----- model -----
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 200, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # registered buffers (no-op because we pre-scale; left for compatibility)
        self.register_buffer("_scaler_shift", torch.full((input_dim,), 0.0, dtype=torch.float32))
        self.register_buffer("_scaler_scale", torch.full((input_dim,), 1.0, dtype=torch.float32))

    @property
    def _is_initialized(self):
        initialized = (torch.isnan(self._scaler_shift) | torch.isnan(self._scaler_scale)).sum() == 0
        initialized &= (self._scaler_scale != 1).all() & (self._scaler_shift != 0).all()
        return initialized

    def initialize_scaler(
        self,
        shift: Union[np.ndarray, torch.Tensor, None] = None,
        scale: Union[np.ndarray, torch.Tensor, None] = None,
    ):
        if self._is_initialized:
            logger.warning("Scaler already initialized. Overwriting the current values.")
        elif shift is not None and scale is not None:
            shift = torch.from_numpy(shift) if isinstance(shift, np.ndarray) else shift
            scale = torch.from_numpy(scale) if isinstance(scale, np.ndarray) else scale
        else:
            msg = """
                shift and scale must be both None
                (falling to default of shift=0.0, scale=1.0)
                or both not None raise ValueError
            """
            logger.error(msg)
            raise ValueError(msg)

        self._scaler_shift.data[:] = shift
        self._scaler_scale.data[:] = scale

    def apply_scaler(self, x):
        return (x - self._scaler_shift.to(x.device)) / self._scaler_scale.to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.apply_scaler(x)
        return self.net(x)

# ----- functions -----
def set_negatives_to_one(tensor):
    # Using torch.where
    return torch.where(tensor < 0, torch.ones_like(tensor), tensor)

def equal_frequency_bins(x, n_bins):
    quantiles = np.linspace(0, 1, n_bins + 1)
    return np.quantile(x, quantiles)

def get_class_weights(
    weights: t.Tensor,
    Y: t.Tensor,
    classes: tuple = (0, 1),
    class_weighted: bool = True,
) -> t.Tensor:

    weights = weights.float()
    Y = Y.long()

    _weights = torch.zeros_like(weights)

    total_weight = weights.sum()

    for _class in classes:
        mask = (Y == _class)
        class_sum = weights[mask].sum()

        if class_sum > 0:
            _weights[mask] = total_weight / class_sum
        else:
            _weights[mask] = 0.0

    return _weights * (weights if class_weighted else 1.0)
# inside training.py

def _predict_in_batches(x: torch.Tensor, model: torch.nn.Module, batch_size: int = 65536) -> torch.Tensor:
    """Run model(x) in batches to keep memory stable. Returns output on x.device."""
    device_model = next(model.parameters()).device
    n = x.shape[0]
    outs = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size].to(device_model, non_blocking=True)
            ob = model(xb)
            outs.append(ob.to(x.device, non_blocking=True))
    return torch.cat(outs, dim=0)

def get_ff_dataset_with_qcd_weights_os(
    dataset: _component_collection,
    model: t.nn.Module,
    qcd_mask_os_loaded: torch.Tensor,
    njets_idx: int = -1,
    njets_groups: Tuple[Tuple[int, ...], ...] = ((0,), (1,), (2, 100)),
    subtract_njets_based: bool = False,
    reweight_njets_based: bool = True,
) -> _component_collection:
    _dataset = deepcopy(dataset)
    _dataset.qcd_weights_os = torch.full_like(
        _dataset.weights.os,
        fill_value=torch.nan
)
    def _leaf_predict(leaf):
        if isinstance(leaf, torch.Tensor):
            return _predict_in_batches(leaf, model, batch_size=65536)
        return leaf
    
    model.eval()
    with torch.no_grad():
        prediction = deepcopy(_dataset.X)
        prediction.ss = _predict_in_batches(_dataset.X.ss, model)
        prediction.os = _predict_in_batches(_dataset.X.os, model)

    # --- save the original QCD masks ---
        qcd_process_mask_ss = _dataset.Y.ss == 2
        qcd_process_mask_os = qcd_mask_os_loaded


    for njets_group in (
        njets_groups
        if subtract_njets_based
        else ((0, 1000),)
    ):
        if len(njets_group) == 1:
            njets_mask_ss = _dataset.X.ss[:, njets_idx] == njets_group[0]
            njets_mask_os = _dataset.X.os[:, njets_idx] == njets_group[0]
        else:
            njets_mask_ss = (_dataset.X.ss[:, njets_idx] >= njets_group[0]) & (_dataset.X.ss[:, njets_idx] <= njets_group[1])
            njets_mask_os = (_dataset.X.os[:, njets_idx] >= njets_group[0]) & (_dataset.X.os[:, njets_idx] <= njets_group[1])

        qcd_mask_ss, non_qcd_mask = njets_mask_ss & qcd_process_mask_ss, njets_mask_ss & ~qcd_process_mask_ss
        qcd_mask_os = njets_mask_os & qcd_process_mask_os


        non_qcd_ss_hist, bins = t.histogram(
            input=prediction.ss[non_qcd_mask],
            bins=t.quantile(prediction.ss[qcd_mask_ss], t.linspace(0, 1, 11)),
            weight=_dataset.weights.ss[non_qcd_mask],
        )

        qcd_weights = _calculate_scaled_event_weights_generalized(
            prediction.ss[qcd_mask_ss].squeeze(),
            _dataset.weights.ss[qcd_mask_ss].squeeze(),
            bins,
            non_qcd_ss_hist,
        )
        _dataset.qcd_weights_os[qcd_mask_os] = qcd_weights

        num_negative_qcd = (qcd_weights < 0).sum().item()

        #logger.info(f"Number of QCD weights < 0: {num_negative_qcd}")
        
        qcd_weights = set_negatives_to_one(qcd_weights)

        _dataset.weights.os[qcd_mask_os] = qcd_weights
        _dataset.class_weights.os[qcd_mask_os] *= _dataset.weights.os[qcd_mask_os]
    # --- store class weights and relabel QCD events to 0 ---
    _dataset.class_weights.os = _dataset.weights.os
    _dataset.Y.os[qcd_mask_os_loaded] = 0  # QCD events become background

    njets_classes = t.zeros_like(qcd_mask_os)
    for idx, njets_group in enumerate(njets_groups if reweight_njets_based else ((0, 1000),)):
        if len(njets_group) == 1:
            njets_mask = _dataset.X.os[:, njets_idx] == njets_group[0]
        else:
            njets_mask = (_dataset.X.os[:, njets_idx] >= njets_group[0]) & (_dataset.X.os[:, njets_idx] <= njets_group[1])
        njets_classes[njets_mask] = idx

    _dataset.class_weights.os = get_class_weights(
        weights=_dataset.weights.os,
        Y=njets_classes,
        classes=tuple(int(it.item()) for it in t.unique(njets_classes)),
        class_weighted=True,
    )

    # --- apply .contiguous() only on tensors ---
    return _dataset.apply_func(lambda x: x.contiguous() if isinstance(x, torch.Tensor) else x)

def _get_backend_and_device(tensor_or_array: Union[np.ndarray, t.Tensor]) -> tuple[Any, Any]:
    if isinstance(tensor_or_array, t.Tensor):
        return t, tensor_or_array.device
    elif isinstance(tensor_or_array, np.ndarray):
        return np, None
    else:
        raise TypeError(f"Input must be a NumPy array or PyTorch tensor, got {type(tensor_or_array)}")

def _calculate_scaled_event_weights_generalized(
    event_values: Union[np.ndarray, t.Tensor],
    event_original_weights: Union[np.ndarray, t.Tensor],
    bins: np.ndarray,
    total_subtraction_per_bin: Union[np.ndarray, t.Tensor],
) -> Union[np.ndarray, t.Tensor]:
    lib, device = _get_backend_and_device(event_values)
    is_torch = (lib == t)
    device_kwargs = {'device': device} if is_torch else {}

    raw = _collection(event_values, event_original_weights, total_subtraction_per_bin)
    
    initial = _collection(
        values=lib.asarray(raw.values, **device_kwargs),
        weights=lib.asarray(raw.weights, **device_kwargs),
        histograms=lib.asarray(raw.histograms, **device_kwargs)
    )
    
    shape_prefix = _collection(
        values=initial.values.shape[:-1],
        weights=initial.weights.shape[:-1],
        histograms=initial.histograms.shape[:-1]
    )

    bins = lib.asarray(bins, dtype=event_values.dtype, **device_kwargs)
    n_bins, n_events = len(bins) - 1, initial.values.shape[-1]

    flat = _collection(
        initial.values.reshape(-1, n_events).contiguous() if is_torch else initial.values.reshape(-1, n_events),
        initial.weights.reshape(-1, n_events),
        initial.histograms.reshape(-1, n_bins)
    )
    batch_size = _collection(
        values=flat.values.shape[0],
        weights=flat.weights.shape[0],
        histograms=flat.histograms.shape[0]
    )

    try:
        common_prefix_dim = np.broadcast_shapes(*shape_prefix.unrolled)
        max_batch_size = int(np.prod(common_prefix_dim)) if common_prefix_dim else 1
    except ValueError as e:
        raise ValueError(f"Prefix shapes {shape_prefix.unrolled} are not broadcastable. Error: {e}")

    if batch_size.values == 1 and max_batch_size > 1:
        flat.values = lib.broadcast_to(flat.values, (max_batch_size, n_events))
    if batch_size.weights == 1 and max_batch_size > 1:
        flat.weights = lib.broadcast_to(flat.weights, (max_batch_size, n_events))
    if batch_size.histograms == 1 and max_batch_size > 1:
        flat.histograms = lib.broadcast_to(flat.histograms, (max_batch_size, n_bins))

    _digitize, digitize_kwargs = (lib.bucketize, {'right': False}) if is_torch else (lib.digitize, {})
    raw_indices = _digitize(flat.values, bins, **digitize_kwargs) - 1

    is_out_of_bounds = (raw_indices < 0) | (raw_indices >= n_bins)
    event_bin_indices = lib.clip(raw_indices, 0, n_bins - 1)

    event_weights_for_summation = flat.weights.clone() if is_torch else flat.weights.copy()
    event_weights_for_summation[is_out_of_bounds] = 0.0  # Zero out weights for out-of-bounds events for sum calculation

    sum_original_weights_per_bin = lib.zeros((max_batch_size, n_bins), dtype=flat.weights.dtype, **device_kwargs)
    if is_torch:
        sum_original_weights_per_bin.scatter_add_(1, event_bin_indices.long(), event_weights_for_summation)
    else:
        for i in range(max_batch_size):
            sum_original_weights_per_bin[i] = lib.bincount(event_bin_indices[i], event_weights_for_summation[i], n_bins)

    scale_factor_per_bin = lib.ones_like(sum_original_weights_per_bin)
    non_zero_sum_mask = sum_original_weights_per_bin != 0

    scale_factor_per_bin[non_zero_sum_mask] = 1.0 - flat.histograms[non_zero_sum_mask] / sum_original_weights_per_bin[non_zero_sum_mask]

    zero_sum_non_zero_subtraction_mask = (sum_original_weights_per_bin == 0) & (flat.histograms != 0)
    scale_factor_per_bin[zero_sum_non_zero_subtraction_mask] = 0.0  # lib.nan

    # Gather Scale Factors for each Event
    if is_torch:
        scale_factors_for_events = lib.gather(scale_factor_per_bin, dim=1, index=event_bin_indices.long())
    else:
        row_idx_gather = lib.arange(max_batch_size)[:, None]
        scale_factors_for_events = scale_factor_per_bin[row_idx_gather, event_bin_indices]

    corrected_event_weights_flat = flat.weights * scale_factors_for_events
    corrected_event_weights_flat[is_out_of_bounds] = flat.weights[is_out_of_bounds]

    return corrected_event_weights_flat.reshape(initial.weights.shape)  # reshape back to original shape

# ------------------
def main():

    # --- load config

    raw = load_config(CONFIG_MODEL_PATH)
    config = Config.from_dict(raw)



    # --- load device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(torch.cuda.get_device_name())

    # --- load data 

    train_pt1 = t.load('data/train1.pt')
    val_pt1 = t.load('data/val1.pt')

    train_pt2 = t.load('data/train2.pt')
    val_pt2 = t.load('data/val2.pt')

    for fold, train_pt, val_pt in zip(['fold1', 'fold2'], [train_pt1, train_pt2], [val_pt1, val_pt2]):

        X_train = train_pt.X.os

        shift = X_train.mean(dim=0).to(device)
        scale  = X_train.std(dim=0, unbiased=False).clamp_min(1e-12).to(device)

        # model, optimizer, scheduler

        model = BinaryClassifier(input_dim=dim, hidden_dim=200, p=0.15).to(device)
        model.initialize_scaler(shift = shift, scale = scale)
        criterion = nn.BCELoss(reduction='none')                                  # there
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            threshold=config.scheduler_threshold,
            threshold_mode='rel',
            cooldown=config.scheduler_cooldown,
            min_lr=config.scheduler_min_lr,
            eps=config.scheduler_eps
        )

        use_amp = (device.type == "cuda") and bool(config.use_amp)
        scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)

        # training loop (full-batch)

        best_val = float('inf')
        counter = 0
        checkpoint = None

        log_rows = []
        logger.info("Start training ")

        njets_groups = ((0,), (1,), (2, 1000))

        qcd_mask_os_train = (train_pt.Y.os == 2)
        qcd_mask_os_val = (val_pt.Y.os == 2)
        for epoch in range(config.n_epochs):

            # ------- update qcd weights ------

            model.eval()
            with torch.no_grad():
                train_pt = get_ff_dataset_with_qcd_weights_os(
                    dataset = train_pt,
                    model = model,
                    qcd_mask_os_loaded = qcd_mask_os_train,
                    njets_idx = 11,
                    njets_groups = ((0,), (1,), (2,100)),
                    subtract_njets_based = True,
                    reweight_njets_based = True,
                )

                val_pt = get_ff_dataset_with_qcd_weights_os(
                    dataset = val_pt,
                    model = model,
                    qcd_mask_os_loaded = qcd_mask_os_val,
                    njets_idx = 11,
                    njets_groups = ((0,), (1,), (2,100)),
                    subtract_njets_based = True,
                    reweight_njets_based = True,
                )





            X_train = train_pt.X.os
            y_train = train_pt.Y.os
            w_train = train_pt.weights.os

            X_val = val_pt.X.os
            y_val = val_pt.Y.os
            w_val = val_pt.weights.os

            dataset_train = TensorDataset(X_train, y_train, w_train)
            dataset_val = TensorDataset(X_val, y_val, w_val)

            train_loader = DataLoader(
                dataset_train,
                batch_size = config.bsize_train,
                shuffle = True,
                drop_last = False
            )

            val_loader = DataLoader(
                dataset_val,
                batch_size = config.bsize_val,
                shuffle= True,
                drop_last = False
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

                with torch.amp.autocast('cuda', enabled=use_amp):
                    # Model now applies StandardScaler inside forward()
                    logits = model(Xb)                   # (B,1)
                    y = yb.float().view(-1, 1)           # targets
                    w = wb.float().view(-1, 1)           # weights

                    # Safety check for BCE
                    if not torch.all((y >= 0) & (y <= 1)):
                        print("BAD TARGETS:", torch.unique(y))
                        raise RuntimeError("Invalid targets for BCE (must be 0/1).")

                    # BCE per sample
                    loss_per_sample = criterion(logits.float(), y)

                    # Weighted loss
                    batch_loss = (loss_per_sample * w).sum()
                    batch_weight = w.sum()

                    loss = batch_loss / batch_weight

                # AMP backward
                scaler_amp.scale(loss).backward()

                # Gradient clipping (AMP‑safe)
                if config.grad_clip and config.grad_clip > 0:
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                scaler_amp.step(optimizer)
                scaler_amp.update()

                # Accumulate epoch totals
                train_loss_sum += batch_loss.item()
                train_weight_sum += batch_weight.item()

            train_loss = train_loss_sum / train_weight_sum



            # ------- VALIDATION -------
            model.eval()
            val_loss_sum = 0.0
            val_weight_sum = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for Xb, yb, wb in val_loader:
                    Xb = Xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    wb = wb.to(device, non_blocking=True)

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        logits = model(Xb)
                        y = yb.float().view(-1, 1)
                        w = wb.float().view(-1, 1)

                        loss_per_sample = criterion(logits.float(), y)
                        batch_loss = (loss_per_sample * w).sum()
                        batch_weight = w.sum()

                        val_loss_sum += batch_loss.item()
                        val_weight_sum += batch_weight.item()

                        # Predictions for accuracy
                        preds = (logits >= 0.5).float()
                        correct += (preds.view(-1) == yb.view(-1)).sum().item()
                        total += yb.numel()

            val_loss = val_loss_sum / val_weight_sum
            val_acc = correct / total
            epoch_time = time.time() - epoch_start



            # ------- LR Scheduler & Logging -------
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(
                f"Epoch {epoch}: train={train_loss:.6f}, "
                f"val={val_loss:.6f}, acc={val_acc:.4f}, LR={current_lr:.6e}"
            )

            log_rows.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
                "time_s": epoch_time,
                "type": "epoch"
            })


            # ----- early stopping -----



            if val_loss + 1e-8 < best_val:
                best_val = val_loss
                counter = 0
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'variables': variables,
                }
            else:
                counter += 1
                if counter >= PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # save checkpoint
        paths_training = StorePathHelper(directory=f"Categorizer_results/inclusive/{fold}")
        if checkpoint is None:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_mean': torch.from_numpy(shift.astype(np.float32)),
                'scaler_scale': torch.from_numpy(scale.astype(np.float32)),
                'variables': variables,
            }
        torch.save(checkpoint, paths_training.autopath.joinpath('model_checkpoint.pth'))
        torch.save(train_pt.qcd_weights_os[qcd_mask_os_train], paths_training.autopath.joinpath('qcd_weights_train.pt'))
        torch.save(val_pt.qcd_weights_os[qcd_mask_os_val], paths_training.autopath.joinpath('qcd_weights_val.pt'))
        
        # save log file
        pd.DataFrame(log_rows).to_pickle(str(paths_training.autopath.joinpath('training_logs.pkl')))

        

# -------------------------------------------

if __name__ == "__main__":
    main()