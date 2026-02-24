import random
import logging
from dataclasses import KW_ONLY, dataclass
import CODE.HELPER as helper
import math
import numpy as np
import pandas as pd
import torch
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt


import matplotlib
from matplotlib.ticker import ScalarFormatter

from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from CustomLogging import setup_logging
#from CustomLogging import setup_logging
from typing import (Any, Callable, Dict, Generator, Iterable, Iterator, List,
                    Optional, Tuple, Type, Union, get_args, get_origin)
from training_wjets import BinaryClassifier
from tap import Tap
from typing import Literal, Generator

# ----- seeds -----

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

# ----- tap ------

class Args(Tap):
    bins: Literal['equi_populated' , 'uniform'] ='equi_populated'

# ----- Constants

INPUT_DIM = 36

MC_PATH   = "../data/MC_data/MC_data.pkl"
DATA_PATH = "../data/MC_data/data.pkl" 


ckpt_pth_fold1 = 'Categorizer_results/QCD/inclusive/fold1/2026-02-19/0_18-59-52/'
ckpt_pth_fold2 = 'Categorizer_results/QCD/inclusive/fold2/2026-02-19/0_19-00-37/'


# ------- lists ----

variables = [
    "pt_1","pt_2","eta_1","eta_2","jpt_1","jpt_2","jeta_1","jeta_2",
    "m_fastmtt","pt_fastmtt","met","njets","mt_tot","m_vis",
    "pt_tt","pt_vis","mjj","pt_dijet","pt_ttjj","deltaEta_jj","deltaR_jj",
    "deltaR_ditaupair","deltaR_1j1","deltaR_1j2",
    "deltaR_2j1","deltaR_2j2","deltaR_12j1","deltaR_12j2","deltaEta_1j1",
    "deltaEta_1j2","deltaEta_2j1","deltaEta_2j2","deltaEta_12j1","deltaEta_12j2", 'tau_decaymode_1', 'tau_decaymode_2',
]

dim = len(variables)


CHECKPOINT_PATH = (
    "../src/Categorizer_results/2026-01-29/0_16-32-09/model_checkpoint.pth"
)

LOG_PATH = (
    "../src/Categorizer_results/2026-01-29/0_16-32-09/training_logs.pkl"
)


# ----- Logging -----

logger = setup_logging(logger=logging.getLogger(__name__))



# ------ config -----


@dataclass
class Config:
    bsize_train: int
    bsize_val: int
    bsize_test: int
    grad_clip: float
    n_epochs: int
    use_amp: bool
    s_scale_max: float
    lr: float

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "Config":
        training  = cfg["training"]
        optimizer = cfg["optimizer"]

        return Config(
            bsize_train=training["bsize_train"],
            bsize_val=training["bsize_val"],
            bsize_test=training["bsize_test"],
            grad_clip=training["grad_clip"],
            n_epochs=training["n_epochs"],
            use_amp=training["use_amp"],
            s_scale_max=training["s_scale_max"],
            lr=optimizer["lr"],
        )


@dataclass
class _same_sign_opposite_sign_split(metaclass=helper.CollectionMeta):
    ss: Union[torch.Tensor, pd.DataFrame, np.ndarray]
    os: Union[torch.Tensor, pd.DataFrame, np.ndarray]


@dataclass
class _component_collection(metaclass=helper.CollectionMeta):
    _: KW_ONLY
    X: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    Y: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    Label: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    weights: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    class_weights: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    process: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None


@dataclass
class _collection:
    values: Any
    weights: Any
    histograms: Any
    
    @property
    def unrolled(self) -> tuple[Any, ...]:
        return (self.values, self.weights, self.histograms)


# ----- model -----


# ------ model utilities -----

def load_model(
    input_dim: int,
    checkpoint_path: str,
    device: torch.device,
) -> BinaryClassifier:
    model = BinaryClassifier(input_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    X = X.to(device, non_blocking=True)
    logits = model(X)
    return logits.squeeze(1).cpu()


# ----- Data utilities -----


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


def split_mc_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trainval, test = train_test_split(
        df,
        test_size=0.5,
        random_state=SEED,
        stratify=df["is_Wjets"]
        
    )
    train, val = train_test_split(
        trainval,
        test_size=0.5,
        random_state=SEED,
        stratify=trainval["is_Wjets"]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trainval, test = train_test_split(
        df, test_size=0.5, random_state=SEED
    )
    train, val = train_test_split(
        trainval, test_size=0.5, random_state=SEED
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def make_tensor_dataset(
    df: pd.DataFrame,
    variables: List[str],
    scaler: StandardScaler,
) -> TensorDataset:
    X = scaler.transform(df[variables].to_numpy(np.float32))
    y = df['is_Wjets'].to_numpy(np.float32)
    w = df['weight'].to_numpy(np.float32)

    return TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y).view(-1, 1),
        torch.from_numpy(w),
    )

def equi_populated_bins(data, n_bins):

    data = np.asarray(data)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(data, quantiles)
    return bin_edges


def mask_DR(df):

    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5))
    mask_a2 = (df.q_1 * df.q_2 > 0)
    mask_a4 = ((df.iso_1 > 0.02) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 < 50)
    mask_DR = (mask_a1 & mask_a2 & mask_a4 & mask_a5 & mask_a6)

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
            os = [],
        )

    return _component_collection(
            X=ss_os_split.apply_func(lambda x: x[training_var].to_numpy(dtype = np.float32)),
            Y=ss_os_split.apply_func(lambda x: x["Label"].to_numpy(dtype = np.float32)),  # or ss_os_split.apply_func(extract_label)
            # instead of _same_sign_opposite_sign_split.apply(lambda x: x["Label"].to_numpy()).to_collection(ss_os_split)
            weights=ss_os_split.apply_func(lambda __df: __df["weight"].to_numpy(dtype = np.float32)),
            class_weights=ss_os_split.apply_func(lambda x: x["class_weights"].to_numpy()),
            process=ss_os_split.apply_func(lambda x: x['process'].to_numpy(dtype = np.float32)),
            Label=ss_os_split.apply_func(lambda x: x["Label"].to_numpy(dtype = np.float32))
        )
def get_my_other_data(df, training_var):
    _df = df  # fold/fold_train/fold_val to load, should contain SS/OS columns
    ss_os_split = _same_sign_opposite_sign_split(
            ss=_df[(_df.SS)],
            os=_df[((_df.OS) & (_df.process == 0))],
        )

    return _component_collection(
            X=ss_os_split.apply_func(lambda x: x[training_var].to_numpy(dtype = np.float32)),
            Y=ss_os_split.apply_func(lambda x: x["Label"].to_numpy(dtype = np.float32)),  # or ss_os_split.apply_func(extract_label)
            # instead of _same_sign_opposite_sign_split.apply(lambda x: x["Label"].to_numpy()).to_collection(ss_os_split)
            weights=ss_os_split.apply_func(lambda __df: __df["weight"].to_numpy(dtype = np.float32)),
            class_weights=ss_os_split.apply_func(lambda x: x["class_weights"].to_numpy()),
            process=ss_os_split.apply_func(lambda x: x['process'].to_numpy(dtype = np.float32)),
            Label=ss_os_split.apply_func(lambda x: x["Label"].to_numpy(dtype = np.float32))
        )
# ----- plotting -----

def CMS_CHANNEL_TITLE(ax, *args, **kwargs):
    ax[0].set_title(
        r"$e\tau_h$",
        fontsize=20,
        loc="left",
        fontproperties="Tex Gyre Heros"
    )

def CMS_NJETS_TITLE(ax, *args, **kwargs):
    ax[0].set_title(
        r"$N_{jets}$ inclusive",
        fontsize=20,
        loc="center",
        fontproperties="Tex Gyre Heros"
    )


def CMS_LUMI_TITLE(ax, *args, **kwargs):
    ax[0].set_title(
        "59.8 $fb^{-1}$ (2018, 13 TeV)",
        fontsize=20,
        loc="right",
        fontproperties="Tex Gyre Heros"
    )


def CMS_LABEL(ax, *args, **kwargs):
    ax[0].text(
        0.025, 0.95,
        "Private work (CMS data/simulation)",
        fontsize=20,
        verticalalignment='top',
        fontproperties="Tex Gyre Heros:italic",
        bbox=dict(facecolor="white", alpha=0, edgecolor="white", boxstyle="round,pad=0.5"),
        transform=ax[0].transAxes
    )


def reorder_for_rowwise_legend(handles, labels, ncol):
    n = len(handles)
    nrows = math.ceil(n / ncol)

    new_handles, new_labels = [], []

    for col in range(ncol):
        for row in range(nrows):
            idx = row * ncol + col
            if idx < n:
                new_handles.append(handles[idx])
                new_labels.append(labels[idx])

    return new_handles, new_labels


def adjust_ylim_for_legend(ax=None, spacing=0.05):
    if ax is None:
        ax = plt.gca()

    fig = ax.figure
    fig.canvas.draw()

    if (leg := ax.get_legend()) is None:
        return

    bbox_leg, bbox_ax = leg.get_window_extent(), ax.get_window_extent()

    legend_height_ratio = bbox_leg.height / bbox_ax.height

    ymin, ymax = ax.get_ylim()
    scale = ax.get_yscale()

    if (available_fraction := 1.0 - legend_height_ratio - spacing) <= 0.1:
        available_fraction = 0.1

    if scale == "linear":
        data_max_y = ax.dataLim.y1
        data_range = data_max_y - ymin
        new_range = data_range / available_fraction
        new_ymax = ymin + new_range
        ax.set_ylim(ymin, new_ymax)

    elif scale == "log":
        log_ymin = np.log10(ymin)
        log_data_max = np.log10(ax.dataLim.y1)
        log_range = log_data_max - log_ymin
        new_log_range = log_range / available_fraction
        new_log_ymax = log_ymin + new_log_range

        new_log_ymax = np.ceil(new_log_ymax)

        new_ymax = 10 ** new_log_ymax
        ax.set_ylim(ymin, new_ymax)

'''
def plot_nn_output(
    outputs_ff: np.ndarray,
    outputs_nonff: np.ndarray,
    outputs_data: np.ndarray,
    weights_ff,
    weights_nff,
    weights_data,
    log_scale: bool,
    filename: str,
):
    bins = np.linspace(0, 1, 11)

    hist_ff, edges = np.histogram(outputs_ff, bins=bins, weights=weights_ff)
    hist_nonff, _  = np.histogram(outputs_nonff, bins=bins, weights=weights_nff)
    hist_data, _   = np.histogram(outputs_data, bins=bins, weights = weights_data)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths  = edges[1:] - edges[:-1]

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        centers, hist_data, np.sqrt(hist_data),
        label="data", marker=".", linestyle="None", color="black"
    )

    plt.bar(
        centers, hist_nonff, width=widths,
        label="non FF-process",
        color="tab:blue", alpha=0.6, edgecolor="black"
    )
    plt.step(
        edges, np.r_[hist_ff, hist_ff[-1]],
        where="post", color="tab:red", lw=2.5,
        label="FF-process"
    )

    if log_scale:
        plt.yscale("log")

    plt.ylim(1e2, None)
    plt.xlabel("NN output")
    plt.ylabel("Counts")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
'''

# ----- main -----

def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model")

    args = Args().parse_args()




    model1 = load_model(INPUT_DIM, ckpt_pth_fold1 + 'model_checkpoint.pth', device)
    model2 = load_model(INPUT_DIM, ckpt_pth_fold2 + 'model_checkpoint.pth', device)

    logger.info("Loading data")

    data_complete = pd.read_feather('../data/data_complete.feather')
    data_DR = mask_DR(data_complete)
    train1, val1, train2, val2 = split_even_odd(data_DR)

    train1 = get_my_data(train1, variables).to_torch(device=None)
    val1   = get_my_data(val1, variables).to_torch(device=None)
    train2 = get_my_data(train2, variables).to_torch(device=None)
    val2   = get_my_data(val2, variables).to_torch(device=None)

    weights_qcd_train1 = torch.load(ckpt_pth_fold1 + 'qcd_weights_qcd_train.pt')
    weights_qcd_val1 = torch.load(ckpt_pth_fold1 + 'qcd_weights_qcd_val.pt')
    weights_qcd_train2 = torch.load(ckpt_pth_fold2 + 'qcd_weights_qcd_train.pt')
    weights_qcd_val2 = torch.load(ckpt_pth_fold2 + 'qcd_weights_qcd_val.pt')

    probs_datat1 = predict_probabilities(model1, train2.X.ss[train2.process.ss == 0], device)
    probs_datav1 = predict_probabilities(model1, val2.X.ss[val2.process.ss == 0], device)
    probs_datat2 = predict_probabilities(model2, train1.X.ss[train1.process.ss == 0], device)
    probs_datav2 = predict_probabilities(model2, val1.X.ss[val1.process.ss == 0], device)

    probs_Wjetst1 = predict_probabilities(model1, train2.X.ss[train2.process.ss == 1], device)
    probs_Wjetsv1 = predict_probabilities(model1, val2.X.ss[val2.process.ss == 1], device)
    probs_Wjetst2 = predict_probabilities(model2, train1.X.ss[train1.process.ss == 1], device)
    probs_Wjetsv2 = predict_probabilities(model2, val1.X.ss[val1.process.ss == 1], device)

    probs_diboson_Jt1 = predict_probabilities(model1, train2.X.ss[train2.process.ss == 2], device)
    probs_diboson_Jv1 = predict_probabilities(model1, val2.X.ss[val2.process.ss == 2], device)
    probs_diboson_Jt2 = predict_probabilities(model2, train1.X.ss[train1.process.ss == 2], device)
    probs_diboson_Jv2 = predict_probabilities(model2, val1.X.ss[val1.process.ss == 2], device)

    probs_diboson_Lt1 = predict_probabilities(model1, train2.X.ss[train2.process.ss == 3], device)
    probs_diboson_Lv1 = predict_probabilities(model1, val2.X.ss[val2.process.ss == 3], device)
    probs_diboson_Lt2 = predict_probabilities(model2, train1.X.ss[train1.process.ss == 3], device)
    probs_diboson_Lv2 = predict_probabilities(model2, val1.X.ss[val1.process.ss == 3], device)

    probs_DYjets_Jt1 = predict_probabilities(model1, train2.X.ss[train2.process.ss == 4], device)
    probs_DYjets_Jv1 = predict_probabilities(model1, val2.X.ss[val2.process.ss == 4], device)
    probs_DYjets_Jt2 = predict_probabilities(model2, train1.X.ss[train1.process.ss == 4], device)
    probs_DYjets_Jv2 = predict_probabilities(model2, val1.X.ss[val1.process.ss == 4], device)

    probs_DYjets_Lt1 = predict_probabilities(model1, train2.X.ss[train2.process.ss == 5], device)
    probs_DYjets_Lv1 = predict_probabilities(model1, val2.X.ss[val2.process.ss == 5], device)
    probs_DYjets_Lt2 = predict_probabilities(model2, train1.X.ss[train1.process.ss == 5], device)
    probs_DYjets_Lv2 = predict_probabilities(model2, val1.X.ss[val1.process.ss == 5], device)

    probs_ST_Jt1 = predict_probabilities(model1, train2.X.ss[train2.process.ss == 6], device)
    probs_ST_Jv1 = predict_probabilities(model1, val2.X.ss[val2.process.ss == 6], device)
    probs_ST_Jt2 = predict_probabilities(model2, train1.X.ss[train1.process.ss == 6], device)
    probs_ST_Jv2 = predict_probabilities(model2, val1.X.ss[val1.process.ss == 6], device)

    probs_ST_Lt1 = predict_probabilities(model1, train2.X.ss[train2.process.ss == 7], device)
    probs_ST_Lv1 = predict_probabilities(model1, val2.X.ss[val2.process.ss == 7], device)
    probs_ST_Lt2 = predict_probabilities(model2, train1.X.ss[train1.process.ss == 7], device)
    probs_ST_Lv2 = predict_probabilities(model2, val1.X.ss[val1.process.ss == 7], device)

    probs_ttbar_Jt1 = predict_probabilities(model1, train2.X.ss[train2.process.ss == 8], device)
    probs_ttbar_Jv1 = predict_probabilities(model1, val2.X.ss[val2.process.ss == 8], device)
    probs_ttbar_Jt2 = predict_probabilities(model2, train1.X.ss[train1.process.ss == 8], device)
    probs_ttbar_Jv2 = predict_probabilities(model2, val1.X.ss[val1.process.ss == 8], device)

    probs_ttbar_Lt1 = predict_probabilities(model1, train2.X.ss[train2.process.ss == 9], device)
    probs_ttbar_Lv1 = predict_probabilities(model1, val2.X.ss[val2.process.ss == 9], device)
    probs_ttbar_Lt2 = predict_probabilities(model2, train1.X.ss[train1.process.ss == 9], device)
    probs_ttbar_Lv2 = predict_probabilities(model2, val1.X.ss[val1.process.ss == 9], device)

    probs_embeddingt1 = predict_probabilities(model1, train2.X.ss[train2.process.ss == 10], device)
    probs_embeddingv1 = predict_probabilities(model1, val2.X.ss[val2.process.ss == 10], device)
    probs_embeddingt2 = predict_probabilities(model2, train1.X.ss[train1.process.ss == 10], device)
    probs_embeddingv2 = predict_probabilities(model2, val1.X.ss[val1.process.ss == 10], device)
                                                              
    probs_nFFt1 = predict_probabilities(model1, train2.X.ss[train2.process.ss > 0], device)
    probs_nFFv1 = predict_probabilities(model1, val2.X.ss[val2.process.ss > 0], device)
    probs_nFFt2 = predict_probabilities(model2, train1.X.ss[train1.process.ss > 0], device)
    probs_nFFv2 = predict_probabilities(model2, val1.X.ss[val1.process.ss > 0], device)

    probs_qcdt1 = predict_probabilities(model1, train2.X.ss[train2.process.ss == 0], device)
    probs_qcdv1 = predict_probabilities(model1, val2.X.ss[val2.process.ss == 0], device)
    probs_qcdt2 = predict_probabilities(model2, train1.X.ss[train1.process.ss == 0], device)
    probs_qcdv2 = predict_probabilities(model2, val1.X.ss[val1.process.ss == 0], device)
    
    

    probs_Wjets = torch.concat([probs_Wjetst1, probs_Wjetsv1, probs_Wjetst2, probs_Wjetsv2], dim = 0).detach().cpu().numpy()
    probs_data = torch.concat([probs_datat1, probs_datav1, probs_datat2, probs_datav2], dim = 0).detach().cpu().numpy()
    probs_diboson_J = torch.concat([probs_diboson_Jt1, probs_diboson_Jv1, probs_diboson_Jt2, probs_diboson_Jv2], dim = 0).detach().cpu().numpy()
    probs_diboson_L = torch.concat([probs_diboson_Lt1, probs_diboson_Lv1, probs_diboson_Lt2, probs_diboson_Lv2], dim = 0).detach().cpu().numpy()
    probs_ST_J = torch.concat([probs_ST_Jt1, probs_ST_Jv1, probs_ST_Jt2, probs_ST_Jv2], dim = 0).detach().cpu().numpy()
    probs_ST_L = torch.concat([probs_ST_Lt1, probs_ST_Lv1, probs_ST_Lt2, probs_ST_Lv2], dim = 0).detach().cpu().numpy()
    probs_DYjets_J = torch.concat([probs_DYjets_Jt1, probs_DYjets_Jv1, probs_DYjets_Jt2, probs_DYjets_Jv2], dim = 0).detach().cpu().numpy()
    probs_DYjets_L = torch.concat([probs_DYjets_Lt1, probs_DYjets_Lv1, probs_DYjets_Lt2, probs_DYjets_Lv2], dim = 0).detach().cpu().numpy()
    probs_ttbar_J = torch.concat([probs_ttbar_Jt1, probs_ttbar_Jv1, probs_ttbar_Jt2, probs_ttbar_Jv2], dim = 0).detach().cpu().numpy()
    probs_ttbar_L = torch.concat([probs_ttbar_Lt1, probs_ttbar_Lv1, probs_ttbar_Lt2, probs_ttbar_Lv2], dim = 0).detach().cpu().numpy()
    probs_embedding = torch.concat([probs_embeddingt1, probs_embeddingv1, probs_embeddingt2, probs_embeddingv2], dim = 0).detach().cpu().numpy()
    probs_nFF = torch.concat([probs_nFFt1, probs_nFFv1, probs_nFFt2, probs_nFFv2], dim = 0).detach().cpu().numpy()

    probs_diboson = np.concatenate([probs_diboson_J, probs_diboson_L], axis = 0)
    probs_ST = np.concatenate([probs_ST_J, probs_ST_L], axis = 0)
    probs_DYjets = np.concatenate([probs_DYjets_J, probs_DYjets_L], axis = 0)
    probs_ttbar = np.concatenate([probs_ttbar_J, probs_ttbar_L], axis = 0)


    probs_qcd = torch.concat([probs_qcdt1, probs_qcdv1, probs_qcdt2, probs_qcdv2], dim = 0).detach().cpu().numpy()

    weights_Wjets = torch.concat(
        [train2.weights.ss[train2.process.ss == 1], val2.weights.ss[val2.process.ss == 1], 
         train1.weights.ss[train1.process.ss == 1], val1.weights.ss[val1.process.ss == 1]], 
         dim = 0).detach().cpu().numpy()
    weights_diboson_J = torch.concat(
        [train2.weights.ss[train2.process.ss == 2], 
             val2.weights.ss[val2.process.ss == 2],
         train1.weights.ss[train1.process.ss == 2], 
             val1.weights.ss[val1.process.ss == 2]], 
         dim = 0).detach().cpu().numpy()
    weights_diboson_L = torch.concat(
        [train2.weights.ss[train2.process.ss == 3], 
             val2.weights.ss[val2.process.ss == 3],
         train1.weights.ss[train1.process.ss == 3], 
             val1.weights.ss[val1.process.ss == 3]], 
         dim = 0).detach().cpu().numpy()
    weights_DYjets_J = torch.concat(
        [train2.weights.ss[train2.process.ss == 4], 
             val2.weights.ss[val2.process.ss == 4],
         train1.weights.ss[train1.process.ss == 4], 
             val1.weights.ss[val1.process.ss == 4]], 
         dim = 0).detach().cpu().numpy()
    weights_DYjets_L = torch.concat(
        [train2.weights.ss[train2.process.ss == 5], 
             val2.weights.ss[val2.process.ss == 5],
         train1.weights.ss[train1.process.ss == 5], 
             val1.weights.ss[val1.process.ss == 5]], 
         dim = 0).detach().cpu().numpy()
    weights_ST_J = torch.concat(
        [train2.weights.ss[train2.process.ss == 6], 
             val2.weights.ss[val2.process.ss == 6],
         train1.weights.ss[train1.process.ss == 6], 
             val1.weights.ss[val1.process.ss == 6]], 
         dim = 0).detach().cpu().numpy()
    weights_ST_L = torch.concat(
        [train2.weights.ss[train2.process.ss == 7],
             val2.weights.ss[val2.process.ss == 7],
         train1.weights.ss[train1.process.ss == 7], 
             val1.weights.ss[val1.process.ss == 7]], 
         dim = 0).detach().cpu().numpy()
    weights_ttbar_J = torch.concat(
        [train2.weights.ss[train2.process.ss == 8], 
             val2.weights.ss[val2.process.ss == 8],
         train1.weights.ss[train1.process.ss == 8], 
             val1.weights.ss[val1.process.ss == 8]], 
         dim = 0).detach().cpu().numpy()
    weights_ttbar_L = torch.concat(
        [train2.weights.ss[train2.process.ss == 9], 
             val2.weights.ss[val2.process.ss == 9],
         train1.weights.ss[train1.process.ss == 9], 
             val1.weights.ss[val1.process.ss == 9]], 
         dim = 0).detach().cpu().numpy()
    weights_embedding = torch.concat(
        [train2.weights.ss[train2.process.ss == 10], 
             val2.weights.ss[val2.process.ss == 10],
         train1.weights.ss[train1.process.ss == 10], 
             val1.weights.ss[val1.process.ss == 10]], 
         dim = 0).detach().cpu().numpy()

    weights_nFF = torch.concat(
        [train2.weights.ss[train2.process.ss > 0], 
             val2.weights.ss[val2.process.ss > 0],
         train1.weights.ss[train1.process.ss > 0], 
             val1.weights.ss[val1.process.ss > 0]], 
         dim = 0).detach().cpu().numpy()


    weights_diboson = np.concatenate([weights_diboson_J, weights_diboson_L], axis = 0)
    weights_DYjets = np.concatenate([weights_DYjets_J, weights_DYjets_L], axis = 0)
    weights_ST = np.concatenate([weights_ST_J, weights_ST_L], axis = 0)
    weights_ttbar = np.concatenate([weights_ttbar_J, weights_ttbar_L], axis = 0)
    
    weights_qcd = torch.concat([weights_qcd_train2, weights_qcd_val2, weights_qcd_train1, weights_qcd_val1], dim = 0).detach().cpu().numpy()

    logger.info(f'len of probs_qcd: {len(probs_qcd)}')
    logger.info(f'len of weights_qcd: {len(weights_qcd)}')

    logger.info(" ------- Plotting NN outputs ------- ")
    logger.info(len(probs_qcd))
    logger.info(len(weights_qcd))

    probs = [probs_qcd, probs_Wjets, probs_embedding, probs_diboson_J, probs_diboson_L, probs_DYjets_J, probs_DYjets_L, probs_ST_J, probs_ST_L, probs_ttbar_J, probs_ttbar_L]
    weights = [weights_qcd, weights_Wjets, weights_embedding, weights_diboson_J, weights_diboson_L, weights_DYjets_J, weights_DYjets_L, weights_ST_J, weights_ST_L, weights_ttbar_J, weights_ttbar_L] 
    labels = ['QCD','Wjets', 'embedding', 'diboson_J', 'diboson_L', 'DYjets_J', 'DYjets_L', 'ST_J', 'ST_L', 'ttbar_J', 'ttbar_L']
    colors = ['#b9ac70', '#e76300', '#ffa90e', '#9f887e', '#94a4a2', '#b9ac70', '#3f90da', '#717581', '#5882ae', '#964c88' ,'#615fc8' ,  '#b9ac70']

    probs_compact = [probs_qcd, probs_Wjets, probs_embedding, probs_diboson, probs_DYjets, probs_ST, probs_ttbar]
    weights_compact = [ weights_qcd, weights_Wjets, weights_embedding, weights_diboson, weights_DYjets, weights_ST, weights_ttbar]
    labels_compact = [r"QCD multijet", r"W+jets", r"$\tau$ embedded", r"Diboson", r"Jet$\rightarrow \tau_{h}$", r"Single t", r'$t\bar{t}$']
    colors_compact = [ '#b9ac70','#e76300', '#ffa90e', '#b9ac70', '#717581', '#717581', '#832db6']
    # -------- calculate 

    if args.bins == 'equi_populated':
        bins = equi_populated_bins(probs_data, 20)
    elif args.bins == 'uniform':
        bins = np.linspace(0, 1, 21)

    bin_widths = np.diff(bins)

    hist_nFF, _ = np.histogram(probs_nFF,weights=weights_nFF, bins= bins)
    QCD_weights = _calculate_scaled_event_weights_generalized(
        event_values = probs_data,
        event_original_weights = np.ones_like(probs_data),
        bins = bins,
        total_subtraction_per_bin=hist_nFF,
    )


    # Add QCD_weights back into data_DR

    # Mask that corresponds exactly to the probs_data events
    mask_qcd_data = (data_DR["SS"] == True) & (data_DR["process"] == 0)

    # Extract row positions inside data_DR
    indices_qcd_DR = data_DR.index[mask_qcd_data].to_numpy()

    # Safety check
    assert len(indices_qcd_DR) == len(QCD_weights), (
        f"Error: DR mask gives {len(indices_qcd_DR)} rows but "
        f"QCD_weights has {len(QCD_weights)} entries"
    )

    # Add new column (NaN everywhere initially)
    data_DR["weight_qcd"] = np.nan
    data_DR.loc[indices_qcd_DR, "weight_qcd"] = QCD_weights


    # Insert qcd_weights into the FULL data_complete

    # Create empty column in full dataset
    data_complete["weight_qcd"] = np.nan

    # Copy values from data_DR into their original row positions
    data_complete.loc[data_DR.index, "weight_qcd"] = data_DR["weight_qcd"]

    # ----------------------------------------------------
    # Save updated file
    # ----------------------------------------------------
    data_complete.reset_index(drop=True).to_feather("../data/data_complete.feather")

    logger.info("Successfully inserted weight_wjets into full data_complete.feather")



    matplotlib.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 1.5,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True
    })


    # ------- plot results ----

    data_counts, bin_edges = np.histogram(
        probs_data, bins=bins
    )
    
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])


    sim_counts, _  = np.histogram(np.concatenate(probs), weights = np.concatenate(weights), bins = bins)

    QCD_counts, _ = np.histogram(probs_qcd, weights = weights_qcd, bins=bins)
    Wjets_counts, _ = np.histogram(probs_Wjets, weights = weights_Wjets, bins = bins)
    embedding_counts, _ = np.histogram(probs_embedding, weights = weights_embedding, bins = bins)
    dibosonJ_counts, _ = np.histogram(probs_diboson_J, weights = weights_diboson_J, bins = bins)
    dibosonL_counts, _ = np.histogram(probs_diboson_L, weights = weights_diboson_L, bins = bins)
    DYjetsJ_counts, _ = np.histogram(probs_DYjets_J, weights = weights_DYjets_J, bins = bins)
    DYjetsL_counts, _ = np.histogram(probs_DYjets_L, weights = weights_DYjets_L, bins = bins)
    STJ_counts, _ = np.histogram(probs_ST_J, weights = weights_ST_J, bins = bins)
    STL_counts, _ = np.histogram(probs_ST_L, weights = weights_ST_L, bins = bins)
    ttbarJ_counts, _ = np.histogram(probs_ttbar_J, weights = weights_ttbar_J, bins = bins)
    ttbarL_counts, _ = np.histogram(probs_ttbar_L, weights = weights_ttbar_L, bins = bins)

    diboson_counts = dibosonJ_counts + dibosonL_counts
    DYjets_counts = DYjetsJ_counts + DYjetsL_counts
    ST_counts = STJ_counts + STL_counts
    ttbar_counts = ttbarJ_counts + ttbarL_counts


    hist_nFF, _ = np.histogram(probs_nFF,weights=weights_nFF, bins= bins)


    QCD_counts_norm = np.divide(QCD_counts, sim_counts)
    Wjets_counts_norm = np.divide(Wjets_counts, sim_counts)
    embedding_counts_norm = np.divide(embedding_counts, sim_counts)
    diboson_counts_norm = np.divide(diboson_counts, sim_counts)
    DYjets_counts_norm = np.divide(DYjets_counts, sim_counts)
    ST_counts_norm = np.divide(ST_counts, sim_counts)
    ttbar_counts_norm = np.divide(ttbar_counts, sim_counts)


    ratio = np.divide(data_counts,
                      sim_counts)


    QCD_counts2, _ = np.histogram(probs_qcd, weights = weights_qcd**2, bins=bins)

    hist_nFF2, _ = np.histogram(probs_nFF,weights=weights_nFF**2, bins= bins)

    y_error = np.sqrt(data_counts)
    x_error = 0.5*bin_widths
    y_error_stat = np.sqrt(QCD_counts2)


    counts_data_reduced, _ = np.histogram(probs_data, weights = QCD_weights, bins = bins)

    fig, ax = plt.subplots(2,1, figsize = (10,8), sharex=True,
        gridspec_kw={'height_ratios': [3,1], 'hspace': 0.05})
    

    CMS_CHANNEL_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_LABEL(ax)

    ax[0].errorbar(bin_centers,counts_data_reduced, yerr = y_error, xerr = x_error, color = 'black', fmt = 'o' ,markersize = 5, label = 'data (reduced)')
    ax[0].bar(bin_centers, QCD_counts, width = bin_widths, color ='#b9ac70', label = 'QCD')
    ax[0].set_ylabel('events')
    ax[0].legend()
    adjust_ylim_for_legend(ax[0])

    ax[1].errorbar(bin_centers, counts_data_reduced/QCD_counts, yerr = y_error/data_counts, xerr = x_error, label = 'ratio', color = 'black', fmt = 'o')
    ax[1].fill_between(
    bin_centers,
    1 - y_error_stat / (Wjets_counts + 1e-10),
    1 + y_error_stat / (Wjets_counts + 1e-10),
    color="gray",
    alpha=0.3,
    step='mid',
    label="stat. unc.")
    
    ax[1].axhline(1, color='red', linestyle='--', linewidth=1.5)
    ax[1].set_ylabel("data / model")
    ax[1].set_ylim([0.5, 1.5])

    fig.savefig('plots/results_data_reduced.png')
    fig.savefig('plots/results_data_reduced.pdf')

    fig, ax = plt.subplots(
        3, 1,
        figsize=(15, 9),
        sharex=True,
        gridspec_kw={'height_ratios': [3,1,1], 'hspace': 0.05}
    )

    CMS_CHANNEL_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_LABEL(ax)


    # X and Y error
    y_error = np.sqrt(data_counts)
    x_error = 0.5*bin_widths
    y_err_stat = np.sqrt(QCD_counts2 + hist_nFF2)

    # --- Upper panel: stacked histograms + data ---
    ax[0].hist(probs_compact, bins=bins, weights=weights_compact, histtype='barstacked',
            label=labels_compact, color=colors_compact)

    ax[0].errorbar(bin_centers, data_counts, yerr=y_error, xerr=x_error,
                fmt='o', color='black', label='data', markersize=5)

    ax[0].set_ylabel("events")

    ax[0].set_ylim([0, 1.4*np.max([np.max(data_counts), np.max(sim_counts)])])
    ax[0].legend(loc='upper right', ncol=3, frameon=False)
    adjust_ylim_for_legend(ax[0])
    # Remove top ticks
    ax[0].tick_params(direction='in', top=True, right=True)

    # --- Lower panel: ratio plot ---
    ax[1].errorbar(bin_centers, ratio, 
                   xerr=x_error,
                   yerr = y_error/data_counts,
                   fmt='o', color='black', markersize=5,
                   label = 'ratio')
    ax[1].fill_between(
    bin_centers,
    1 - y_err_stat / (Wjets_counts + QCD_counts + hist_nFF + 1e-10),
    1 + y_err_stat / (Wjets_counts + QCD_counts + hist_nFF + 1e-10),
    color="gray",
    alpha=0.3,
    step='mid',
    label="stat. unc.")
    ax[1].axhline(1, color='red', linestyle='--', linewidth=1.5)
    ax[1].set_ylabel("data / sim")
    ax[1].set_ylim([0.5, 1.5])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc = 'upper right', ncol = 2)
    
    ax[2].bar(bin_centers, QCD_counts_norm, color =  '#b9ac70', width = bin_widths )
    ax[2].bar(bin_centers, Wjets_counts_norm, bottom = QCD_counts_norm, color ='#e76300', width = bin_widths)
    ax[2].bar(bin_centers, embedding_counts_norm, bottom = QCD_counts_norm + Wjets_counts_norm, color = '#ffa90e', width = bin_widths)
    ax[2].bar(bin_centers, diboson_counts_norm, bottom = QCD_counts_norm + Wjets_counts_norm + embedding_counts_norm, color = '#94a4a2', width = bin_widths)
    ax[2].bar(bin_centers, DYjets_counts_norm, bottom = QCD_counts_norm + Wjets_counts_norm + embedding_counts_norm + diboson_counts_norm, color = '#b9ac70', width = bin_widths)
    ax[2].bar(bin_centers, ST_counts_norm, bottom = QCD_counts_norm + Wjets_counts_norm + embedding_counts_norm + diboson_counts_norm + DYjets_counts_norm, color = '#717581', width = bin_widths)
    ax[2].bar(bin_centers, ttbar_counts_norm, bottom = QCD_counts_norm + Wjets_counts_norm + embedding_counts_norm + diboson_counts_norm + DYjets_counts_norm + ST_counts_norm, color = '#832db6', width = bin_widths)
    ax[2].set_xlabel("NN output")
    ax[2].set_ylabel('proc. frac.')
    #ax[2].set_ylim([0,1])
    # Tight layout

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    if args.bins == 'equi_populated':
        fig.savefig(f'plots/results_training_equi_QCD.png')
        fig.savefig(f'plots/results_training_equi_QCD.pdf')
    elif args.bins == 'uniform':
        fig.savefig(f'plots/results_training_uniform_QCD.png')
        fig.savefig(f'plots/results_training_uniform_QCD.pdf')




# ---------------

if __name__ == "__main__":
    main()