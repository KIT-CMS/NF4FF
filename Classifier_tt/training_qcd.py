from copy import deepcopy
import logging
from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import time
from typing import Any, Dict, List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tap import Tap
import torch
import torch as t
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import classes.helper as helper
from classes.Logging import setup_logging
from classes.models import BinaryClassifier
from classes.path_managment import StorePathHelper
from classes.Collection import load_config


# ----- seed -----

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
t.set_num_threads(8)
SEED = 42

# ------ logger -----

logger = setup_logging(logger=logging.getLogger(__name__))

# ----- TAP Arguments -----
class Args(Tap):
    loc: Literal["remote", "present"] = "present"
    embedding: Literal["embedding", "no_embedding"] = "no_embedding"

# ----- constants ------
PATIENCE = 20
QCD_WEIGHT_BINNING = 'dynamic'
QCD_WEIGHT_N_BINS = 20
QCD_WEIGHT_DYNAMIC_DELTA = 4.0
QCD_WEIGHT_DYNAMIC_DELTA_LAST = 4.0
QCD_WEIGHT_DYNAMIC_MIN_QCD_YIELD = 10.0

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
    process: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    qcd_weights: Union[t.Tensor, None] = None
    SR_like: Union [t.Tensor, int, None] = None

@dataclass
class _collection:
    values: Any
    weights: Any
    histograms: Any
    
    @property
    def unrolled(self) -> tuple[Any, ...]:
        return (self.values, self.weights, self.histograms)


# ----- functions -----

@torch.no_grad()
def predict_probabilities(model: nn.Module, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    '''
    Returns NN outputs
    '''

    X = X.to(device, non_blocking=True)
    logits = model(X)
    return logits.squeeze(1).cpu()

def set_negatives_to_one(tensor):
    return torch.where(tensor < 0, torch.zeros_like(tensor), tensor)

def equal_frequency_bins(x, n_bins):
    quantiles = np.linspace(0, 1, n_bins + 1)
    return np.quantile(x, quantiles)


def find_dynamic_bin_edges(
    values_A: Union[np.ndarray, t.Tensor],
    weights_A: Union[np.ndarray, t.Tensor],
    values_B: Union[np.ndarray, t.Tensor],
    weights_B: Union[np.ndarray, t.Tensor],
    delta: float = 0.0,
    delta_last: float = 0.0,
    min_A_yield: float = 0.0,
    max_val: float = 1.0,
    min_val: float = 0.0,
) -> Union[np.ndarray, t.Tensor]:
    """
    Finds bin edges dynamically by walking from max_val down to min_val.
    Ensures that for each bin: Sum(w_A) >= min_A_yield AND Sum(w_A) - Sum(w_B) > delta.
    Compatible with both NumPy arrays and PyTorch tensors.
    """

    is_torch = isinstance(values_A, t.Tensor)

    if is_torch:
        values_all = t.cat([values_A, values_B])
        weights_net = t.cat([weights_A, -weights_B])
        weights_A_only = t.cat([weights_A, t.zeros_like(weights_B)])

        sort_idx = t.argsort(values_all, descending=True)

    else:
        values_all = np.concatenate([values_A, values_B])
        weights_net = np.concatenate([weights_A, -weights_B])
        weights_A_only = np.concatenate([weights_A, np.zeros_like(weights_B)])

        sort_idx = np.argsort(values_all)[::-1]

    values_sorted = values_all[sort_idx]
    weights_net_sorted = weights_net[sort_idx]
    weights_A_only_sorted = weights_A_only[sort_idx]

    values_sorted_list = values_sorted.tolist()
    weights_net_sorted_list = weights_net_sorted.tolist()
    weights_A_only_sorted_list = weights_A_only_sorted.tolist()

    # Top-Down Cumulative Sum Walking
    edges, accumulative_net, accumulative_A = [max_val], 0.0, 0.0
    for i in range(len(values_sorted_list)):
        if values_sorted_list[i] < min_val:
            break

        accumulative_net += weights_net_sorted_list[i]
        accumulative_A += weights_A_only_sorted_list[i]

        if accumulative_A >= min_A_yield and accumulative_net > delta:
            edges.append(values_sorted_list[i])
            accumulative_net, accumulative_A = 0.0, 0.0

    edges.append(min_val)

    # Upward merging if needed
    while len(edges) > 2:
        low, high = edges[-1], edges[-2]

        if is_torch:
            mask = (values_all >= low) & (values_all < high)
            final_bin_net = t.sum(weights_net[mask]).item()
        else:
            mask = (values_all >= low) & (values_all < high)
            final_bin_net = np.sum(weights_net[mask])

        if final_bin_net > delta_last:
            break

        edges.pop(-2)

    edges.reverse()

    if is_torch:
        return t.tensor(edges, dtype=values_A.dtype, device=values_A.device)
    else:
        return np.array(edges, dtype=values_A.dtype)


def build_qcd_weight_bins(
    qcd_values: t.Tensor,
    qcd_weights: t.Tensor,
    non_qcd_values: t.Tensor,
    non_qcd_weights: t.Tensor,
    binning: Literal['quantile', 'dynamic'] = 'dynamic',
    n_bins: int = 20,
    dynamic_delta: float = 100.0,
    dynamic_delta_last: float = 100.0,
    dynamic_min_qcd_yield: float = 100.0,
) -> t.Tensor:
    
    if binning == 'quantile':
        bins = t.quantile(qcd_values, t.linspace(0, 1, n_bins + 1, device=qcd_values.device))
    elif binning == 'dynamic':
        min_val = t.minimum(qcd_values.min(), non_qcd_values.min()).item()
        max_val = t.maximum(qcd_values.max(), non_qcd_values.max()).item()
        bins = find_dynamic_bin_edges(
            values_A=qcd_values,
            weights_A=qcd_weights,
            values_B=non_qcd_values,
            weights_B=non_qcd_weights,
            delta=dynamic_delta,
            delta_last=dynamic_delta_last,
            min_A_yield=dynamic_min_qcd_yield,
            min_val=min_val,
            max_val=max_val,
        )
    else:
        raise ValueError(f"Unknown qcd binning option: {binning}")

    bins = t.unique(bins, sorted=True)
    if bins.numel() < 2:
        logger.warning("QCD bin builder returned <2 unique edges. Falling back to quantile binning.")
        bins = t.quantile(qcd_values, t.linspace(0, 1, n_bins + 1, device=qcd_values.device))
        bins = t.unique(bins, sorted=True)

    if bins.numel() < 2:
        raise RuntimeError("Could not construct valid QCD bin edges.")

    return bins


def get_ff_dataset_with_qcd_weights_ss(
    dataset: _component_collection,
    model: t.nn.Module,
    qcd_process_mask_ss_loaded: torch.Tensor,
    device,
    njets_idx: int,
    njets_groups: Tuple[Tuple[int, ...], ...] = ((0,), (1,), (2, 100)),
    subtract_njets_based: bool = False,
    qcd_weight_binning: Literal['quantile', 'dynamic'] = 'quantile',
    qcd_weight_n_bins: int = 20,
    qcd_weight_dynamic_delta: float = 100.0,
    qcd_weight_dynamic_delta_last: float = 100.0,
    qcd_weight_dynamic_min_qcd_yield: float = 100.0,

) -> _component_collection:
    """
    Build a dataset where QCD weights are computed *only in the SS region*
    and saved to dataset.weights.ss.

    Differences from OS version:
    - Only SS quantities are used.
    - SR_like.ss determines the shape slices.
    - QCD weights are saved into weights.ss (and class_weights.ss if present).
    - No OS manipulation, no label rewriting.
    """

    _dataset = deepcopy(dataset)

    # Basic validation
    if not hasattr(_dataset, "SR_like") or not hasattr(_dataset.SR_like, "ss"):
        raise AttributeError("Expected dataset.SR_like.ss boolean tensor.")

    # Container for QCD weights in SS
    _dataset.qcd_weights_ss = torch.full_like(
        _dataset.weights.ss,
        fill_value=torch.nan,
    )

    # --- predictions ---
    model.eval()
    with torch.no_grad():
        prediction_ss = predict_probabilities(model, _dataset.X.ss, device)

    # --- masks ---
    qcd_mask_ss = qcd_process_mask_ss_loaded       # provided QCD-like SS mask
    non_qcd_mask_ss = ~qcd_mask_ss                 # everything else in SS

    # Loop over njets groups (or inclusive if not subtracting)
    for njets_group in (njets_groups if subtract_njets_based else ((0, 1000),)):
        # Define njets mask
        if len(njets_group) == 1:
            njets_mask_ss = _dataset.X.ss[:, njets_idx] == njets_group[0]
        else:
            njets_mask_ss = (
                (_dataset.X.ss[:, njets_idx] >= njets_group[0]) &
                (_dataset.X.ss[:, njets_idx] <= njets_group[1])
            )

        qcd_mask = qcd_mask_ss & njets_mask_ss
        non_qcd_mask = non_qcd_mask_ss & njets_mask_ss

        # --- split by SR_like.ss ---
        for sr_value in (True, False):
            sr_mask = (_dataset.SR_like.ss == sr_value)

            qcd_mask_sr = qcd_mask & sr_mask
            non_qcd_mask_sr = non_qcd_mask & sr_mask

            # Skip invalid regions
            if (
                qcd_mask_sr.sum() == 0
                or non_qcd_mask_sr.sum() == 0
            ):
                continue

            bins = build_qcd_weight_bins(
                qcd_values=prediction_ss[qcd_mask_sr].squeeze(),
                qcd_weights=_dataset.weights.ss[qcd_mask_sr].squeeze(),
                non_qcd_values=prediction_ss[non_qcd_mask_sr].squeeze(),
                non_qcd_weights=_dataset.weights.ss[non_qcd_mask_sr].squeeze(),
                binning=qcd_weight_binning,
                n_bins=qcd_weight_n_bins,
                dynamic_delta=qcd_weight_dynamic_delta,
                dynamic_delta_last=qcd_weight_dynamic_delta_last,
                dynamic_min_qcd_yield=qcd_weight_dynamic_min_qcd_yield,
            )

            logger.info(
                "QCD weight bins (%s, njets=%s, SR_like=%s): %d",
                qcd_weight_binning,
                njets_group,
                sr_value,
                max(int(bins.numel()) - 1, 0),
            )

            non_qcd_hist, bins = t.histogram(
                input=prediction_ss[non_qcd_mask_sr],
                bins=bins,
                weight=_dataset.weights.ss[non_qcd_mask_sr],
            )

            # Compute QCD weights
            qcd_weights = _calculate_scaled_event_weights_generalized(
                prediction_ss[qcd_mask_sr].squeeze(),
                t.ones_like(prediction_ss[qcd_mask_sr].squeeze()),
                bins,
                non_qcd_hist,
            )

            qcd_weights = set_negatives_to_one(qcd_weights)

            # Save weights
            _dataset.weights.ss[qcd_mask_sr] = qcd_weights
            if hasattr(_dataset, "class_weights"):
                _dataset.class_weights.ss[qcd_mask_sr] *= qcd_weights

    return _dataset.apply_func(
        lambda x: x.contiguous() if isinstance(x, torch.Tensor) else x
    )



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


# DR_mask

def mask_DR(df):

    mask_a1 = (df.q_1 * df.q_2 > 0)
    mask_a2 = ((df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a3 = ((df.id_tau_vsJet_VLoose_1 > 0.5))
    mask_a4 = ((df.id_tau_vsJet_VLoose_2 > 0.5))
    mask_DR = (mask_a1 & mask_a2 & mask_a3 & mask_a4)

    return df[mask_DR].copy()


def split_even_odd(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold1 = df[df.event_var  == 0].reset_index(drop=True)
    fold2 = df[df.event_var == 1].reset_index(drop=True)

    train1, val1 = train_test_split(
        fold1, test_size=0.5, random_state=SEED
    )
    train2, val2 = train_test_split(
        fold2, test_size=0.5, random_state=SEED
    )

    return train1.reset_index(drop=True), val1.reset_index(drop=True), train2.reset_index(drop=True), val2.reset_index(drop=True)

def get_my_data(df, training_var):
    _df = df  # fold/fold_train/fold_val to load, should contain SS/OS columns
    ss_os_split = _same_sign_opposite_sign_split(
            ss=_df[(_df.SS)],
            os= [],
        )

    return _component_collection(
            X=ss_os_split.apply_func(lambda x: x[training_var].to_numpy(dtype = np.float32)),
            Y=ss_os_split.apply_func(lambda x: x["Label"].to_numpy(dtype = np.float32)),  # or ss_os_split.apply_func(extract_label)
            weights=ss_os_split.apply_func(lambda __df: __df["weight"].to_numpy(dtype = np.float32)),
            class_weights=ss_os_split.apply_func(lambda x: x["class_weights"].to_numpy()),
            process=ss_os_split.apply_func(lambda x: x['process'].to_numpy(dtype = np.float32)),
            SR_like = ss_os_split.apply_func(lambda x: x["id_tau_vsJet_Tight_2"].to_numpy(dtype=np.float32)),
        )


def update_last_training_folder(run_dir: Union[str, Path], last_dir: Union[str, Path]) -> None:
    run_dir = Path(run_dir)
    last_dir = Path(last_dir)

    if not run_dir.exists():
        logger.warning("Run directory does not exist, cannot update latest folder: %s", run_dir)
        return

    if last_dir.exists():
        shutil.rmtree(last_dir)

    last_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, last_dir)
    logger.info("Updated latest training folder: %s", last_dir)

# ---------------------------------------

def main():
       # --- load config
    args = Args().parse_args()
    
    if args.loc == "remote":
        CONFIG_MODEL_PATH = '/run/user/1003/gvfs/sftp:host=portal1.etp.kit.edu,user=tapp/work/tapp/TauFF/NF4FF/Classifier_tt/configs/config_NN.yaml'
        CONFIG_SETTINGS_PATH = '/run/user/1003/gvfs/sftp:host=portal1.etp.kit.edu,user=tapp/work/tapp/TauFF/NF4FF/Classifier_tt/configs/config_settings.yaml'
    elif args.loc == "present":
        CONFIG_MODEL_PATH = '/work/tapp/TauFF/NF4FF/Classifier_tt/configs/config_NN.yaml'
        CONFIG_SETTINGS_PATH = '/work/tapp/TauFF/NF4FF/Classifier_tt/configs/config_settings.yaml'
    else:
        logger.error("Invalid location argument: %s", args.loc)
        exit()

    raw = load_config(CONFIG_MODEL_PATH)
    config = Config.from_dict(raw)
    cfg = load_config(CONFIG_SETTINGS_PATH)


    # -----load variables from settings config

    variables = cfg["variables"]
    dim = len(variables)


    # --- load device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")
    logger.info(
        "QCD weight binning: %s (n_bins=%d, delta=%.1f, delta_last=%.1f, min_qcd_yield=%.1f)",
        QCD_WEIGHT_BINNING,
        QCD_WEIGHT_N_BINS,
        QCD_WEIGHT_DYNAMIC_DELTA,
        QCD_WEIGHT_DYNAMIC_DELTA_LAST,
        QCD_WEIGHT_DYNAMIC_MIN_QCD_YIELD,
    )

    if device.type == "cuda":
        logger.info(torch.cuda.get_device_name())
    

    # --- load data 

    data_complete = pd.read_feather(cfg["paths"]["input_dir"][args.loc] + args.embedding + "/combined_data.feather")
    data_DR = mask_DR(data_complete)


    data_DR.loc[data_DR['process'] != 0, 'Label'] = 0
    data_DR.loc[data_DR['process'] == 0, 'Label'] = 1


    train1, val1, train2, val2 = split_even_odd(data_DR)

    train_pt1 = get_my_data(train1, variables).to_torch(device=None)
    val_pt1   = get_my_data(val1, variables).to_torch(device=None)
    train_pt2 = get_my_data(train2, variables).to_torch(device=None)
    val_pt2   = get_my_data(val2, variables).to_torch(device=None)


    for fold, train_pt, val_pt  in zip(['fold1', 'fold2'], [train_pt1, train_pt2], [val_pt1, val_pt2]):

        X_train = train_pt.X.ss

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
        qcd_mask_ss_train = (train_pt.process.ss == 0)
        qcd_mask_ss_val = (val_pt.process.ss == 0)


        for epoch in range(config.n_epochs):

            # ------- update qcd weights (every 5 epochs) ------

            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    train_pt = get_ff_dataset_with_qcd_weights_ss(
                        dataset = train_pt,
                        model = model,
                        qcd_process_mask_ss_loaded= qcd_mask_ss_train,
                        device = device,
                        njets_idx = variables.index("njets"),
                        njets_groups = ((0,), (1,), (2,100)),
                        subtract_njets_based = True,
                        qcd_weight_binning = QCD_WEIGHT_BINNING,
                        qcd_weight_n_bins = QCD_WEIGHT_N_BINS,
                        qcd_weight_dynamic_delta = QCD_WEIGHT_DYNAMIC_DELTA,
                        qcd_weight_dynamic_delta_last = QCD_WEIGHT_DYNAMIC_DELTA_LAST,
                        qcd_weight_dynamic_min_qcd_yield = QCD_WEIGHT_DYNAMIC_MIN_QCD_YIELD,
                    )

                    val_pt = get_ff_dataset_with_qcd_weights_ss(
                        dataset = val_pt,
                        model = model,
                        qcd_process_mask_ss_loaded=qcd_mask_ss_val,
                        device = device,
                        njets_idx = variables.index("njets"),
                        njets_groups = ((0,), (1,), (2,100)),
                        subtract_njets_based = True,
                        qcd_weight_binning = QCD_WEIGHT_BINNING,
                        qcd_weight_n_bins = QCD_WEIGHT_N_BINS,
                        qcd_weight_dynamic_delta = QCD_WEIGHT_DYNAMIC_DELTA,
                        qcd_weight_dynamic_delta_last = QCD_WEIGHT_DYNAMIC_DELTA_LAST,
                        qcd_weight_dynamic_min_qcd_yield = QCD_WEIGHT_DYNAMIC_MIN_QCD_YIELD,
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
        paths_training = StorePathHelper(directory=f"results/QCD/inclusive/{fold}")
        
        # Ensure QCD weights are computed one final time before saving
        model.eval()
        with torch.no_grad():
            train_pt = get_ff_dataset_with_qcd_weights_ss(
                dataset = train_pt,
                model = model,
                qcd_process_mask_ss_loaded=qcd_mask_ss_train,
                device = device,
                njets_idx = variables.index("njets"),
                njets_groups = ((0,), (1,), (2,100)),
                subtract_njets_based = True,
                qcd_weight_binning = QCD_WEIGHT_BINNING,
                qcd_weight_n_bins = QCD_WEIGHT_N_BINS,
                qcd_weight_dynamic_delta = QCD_WEIGHT_DYNAMIC_DELTA,
                qcd_weight_dynamic_delta_last = QCD_WEIGHT_DYNAMIC_DELTA_LAST,
                qcd_weight_dynamic_min_qcd_yield = QCD_WEIGHT_DYNAMIC_MIN_QCD_YIELD,
            )

            val_pt = get_ff_dataset_with_qcd_weights_ss(
                dataset = val_pt,
                model = model,
                qcd_process_mask_ss_loaded=qcd_mask_ss_val,
                device = device,
                njets_idx = variables.index("njets"),
                njets_groups = ((0,), (1,), (2,100)),
                subtract_njets_based = True,
                qcd_weight_binning = QCD_WEIGHT_BINNING,
                qcd_weight_n_bins = QCD_WEIGHT_N_BINS,
                qcd_weight_dynamic_delta = QCD_WEIGHT_DYNAMIC_DELTA,
                qcd_weight_dynamic_delta_last = QCD_WEIGHT_DYNAMIC_DELTA_LAST,
                qcd_weight_dynamic_min_qcd_yield = QCD_WEIGHT_DYNAMIC_MIN_QCD_YIELD,
            )
        
        if checkpoint is None:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_mean': torch.from_numpy(shift.astype(np.float32)),
                'scaler_scale': torch.from_numpy(scale.astype(np.float32)),
                'variables': variables,
            }

        probs_data = predict_probabilities(model, val_pt.X.ss[val_pt.process.ss == 0], device)
        probs_nqcd = predict_probabilities(model, val_pt.X.ss[val_pt.process.ss > 0], device)
        weights_qcd = val_pt.weights.ss[val_pt.process.ss == 0].to(device).detach().cpu().numpy()
        weights_nqcd = val_pt.weights.ss[val_pt.process.ss > 0].to(device).detach().cpu().numpy()
        bins = np.quantile(
                    probs_data,
                    np.linspace(0, 1, 21))
        bin_widths = np.diff(bins)
        data_counts, bin_edges = np.histogram(probs_data, bins=bins)
        qcd_counts, _ = np.histogram(probs_data, weights = weights_qcd, bins=bins)
        nqcd_counts, _ = np.histogram(probs_nqcd, weights = weights_nqcd, bins=bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.errorbar(bin_centers,data_counts,  color = 'black', fmt = 'o' ,markersize = 5, label = 'data (reduced)')
        plt.bar(bin_centers, qcd_counts, width = bin_widths, color ='#b9ac70', label = 'QCD')
        plt.bar(bin_centers, nqcd_counts, bottom = qcd_counts, color = 'grey', width = bin_widths, label = 'nQCD')
        plt.legend()
        plt.savefig(f'closure_plot_{fold}.png')
        plt.close()

        plt.hist(probs_data, bins = bins, color = 'black', alpha = 0.3) 
        plt.hist([])

        torch.save(checkpoint, paths_training.autopath.joinpath('model_checkpoint.pth'))
        torch.save(train_pt.weights.ss[qcd_mask_ss_train], paths_training.autopath.joinpath('qcd_weights_qcd_train.pt'))
        torch.save(val_pt.weights.ss[qcd_mask_ss_val], paths_training.autopath.joinpath('qcd_weights_qcd_val.pt'))
        
        # save log file
        pd.DataFrame(log_rows).to_pickle(str(paths_training.autopath.joinpath('training_logs.pkl')))

        run_dir = Path(str(paths_training.autopath))
        last_dir = Path("results") / "QCD" / "inclusive" / fold / "last"
        update_last_training_folder(run_dir=run_dir, last_dir=last_dir)

        

# -------------------------------------------

if __name__ == "__main__":
    main()