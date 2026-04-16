import torch as t
import torch.nn as nn
import pandas as pd
import numpy as np
from Classifier.plot_training_results_wjets_no_process_fraction import Config
from classes.helper import _same_sign_opposite_sign_split, _component_collection
from classes.models import BinaryClassifier, FoldCombinedDNN_
from sklearn.model_selection import train_test_split
from classes.config_loader import load_config
from dataclasses import dataclass
from typing import Dict, Any

SEED = 42
t.manual_seed(SEED)
np.random.seed(SEED)

# ----- constants -----

CONFIG_MODEL_PATH = 'configs/config_NN.yaml'


# ----- variables -----

variables = [
    "pt_1","pt_2","eta_1","eta_2","jpt_1","jpt_2","jeta_1","jeta_2",
    "m_fastmtt","pt_fastmtt","met","njets","mt_tot","m_vis",
    "pt_tt","pt_vis","mjj","pt_dijet","pt_ttjj","deltaEta_jj","deltaR_jj",
    "deltaR_ditaupair","deltaR_1j1","deltaR_1j2",
    "deltaR_2j1","deltaR_2j2","deltaR_12j1","deltaR_12j2","deltaEta_1j1",
    "deltaEta_1j2","deltaEta_2j1","deltaEta_2j2","deltaEta_12j1","deltaEta_12j2", 'tau_decaymode_1', 'tau_decaymode_2', 'nbtag',
]

dim = len(variables)


# ----- utilities -----

@dataclass
class Config_binary_classifier:
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
    def from_dict(cfg: Dict[str, Any]) -> "Config_binary_classifier":
        training = cfg["training"]
        optimizer = cfg["optimizer"]
        scheduler = cfg["scheduler"]

        return Config_binary_classifier(
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
            scheduler_eps=scheduler["eps"],)

def get_my_data(df, training_var):

    ss_region = df[df.SS]
    os_region = df[(df.OS & (df.Label != 2)) | (df.SS & (df.Label == 2))]

    ss_os_split = _same_sign_opposite_sign_split(
        ss = ss_region,
        os = os_region
    )

    return _component_collection(
        X = ss_os_split.apply_func(lambda x: x[training_var].to_numpy(dtype=np.float32)),
        Y = ss_os_split.apply_func(lambda x: x["Label"].to_numpy(dtype=np.float32)),
        weights = ss_os_split.apply_func(lambda x: x["weight"].to_numpy(dtype=np.float32)),
        class_weights = ss_os_split.apply_func(lambda x: x["class_weights"].to_numpy()),
        process = ss_os_split.apply_func(lambda x: x["process"].to_numpy(dtype=np.float32)),
        event_var = ss_os_split.apply_func(lambda x: x["event_var"].to_numpy(dtype=np.float32)),
    )


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

    weights_sorted = values_all[sort_idx]
    weights_net_sorted = weights_net[sort_idx]
    weights_A_only_sorted = weights_A_only[sort_idx]

    values_sorted_list = weights_sorted.tolist()
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
    binning: Literal['quantile', 'dynamic'] = 'quantile',
    n_bins: int = 10,
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



# ------ masks -----

def mask_DR(df):

    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 > 70)
    mask_DR = (mask_a1 & mask_a2  & mask_a4 & mask_a5 & mask_a6)

    return df[mask_DR].copy()

# ------ main ------

def main():

    raw = load_config(CONFIG_MODEL_PATH)
    config = Config_binary_classifier.from_dict(raw)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")    

    data_complete = pd.read_feather('data/data_complete.feather')
    data_DR = mask_DR(data_complete)
    
    train, val = train_test_split(data_DR, test_size=0.5, random_state=SEED)

    train = get_my_data(train, training_var=variables)
    val = get_my_data(val, training_var=variables)

    X_train, Y_train, W_train = train.X.os, train.Y.os, train.weights.os
    X_val, Y_val, W_val = val.X.os, val.Y.os, val.weights.os

    shift = X_train.mean(dim=0).to(device)
    scale = X_train.std(dim=0, unbiased=False).clamp_min(1e-12).to(device)

    even_model = BinaryClassifier(input_dim=dim, hidden_dim=200, p=0.15).to(device)
    odd_model = BinaryClassifier(input_dim=dim, hidden_dim=200, p=0.15).to(device)
    even_model.initialize_scaler(shift=shift, scale=scale)
    odd_model.initialize_scaler(shift=shift, scale=scale)
    
    model = FoldCombinedDNN_(
        even_model=even_model,
        odd_model=odd_model
    ).to(device)

    criterion = nn.BCELoss(reduction="none")
    optimizer = t.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        threshold=config.scheduler_threshold,
        threshold_mode="rel",
        cooldown=config.scheduler_cooldown,
        min_lr=config.scheduler_min_lr,
        eps=config.scheduler_eps,
    )

    qcd_mask_train = (Y_train == 0)
    qcd_mask_val = (Y_val == 0)

    for epoch in range(1, config.n_epochs + 1):

        model.eval()


if __name__ == "__main__":
    main()