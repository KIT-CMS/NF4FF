import random
import logging
from dataclasses import KW_ONLY, dataclass
import CODE.HELPER as helper

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from CustomLogging import setup_logging
from typing import (Any, Callable, Dict, Generator, Iterable, Iterator, List,
                    Optional, Tuple, Type, Union, get_args, get_origin)
from tap import Tap
from typing import Literal, Generator

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# ----- seeds -----

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ----- tap ------

class Args(Tap):
    njets: Literal['inclusive', 'njets0', 'njets1', 'njets2']



# ----- Constants

INPUT_DIM = 20


ckpt_pth_inclusive_fold1 = 'Categorizer_results/inclusive/pretraining/fold1/2026-02-05/0_18-31-18/model_checkpoint.pth'
ckpt_pth_inclusive_fold2 = 'Categorizer_results/inclusive/pretraining/fold2/2026-02-05/0_18-40-49/model_checkpoint.pth'

ckpt_pth_njets0_fold1 = 'Categorizer_results/njets0/pretraining/fold1/2026-02-03/0_17-23-17/model_checkpoint.pth'
ckpt_pth_njets0_fold2 = 'Categorizer_results/njets0/pretraining/fold2/2026-02-03/0_17-24-54/model_checkpoint.pth'

ckpt_pth_njets1_fold1 = 'Categorizer_results/njets1/pretraining/fold1/2026-02-03/0_17-25-43/model_checkpoint.pth'
ckpt_pth_njets1_fold2 = 'Categorizer_results/njets1/pretraining/fold2/2026-02-03/0_17-26-38/model_checkpoint.pth'

ckpt_pth_njets2_fold1 = 'Categorizer_results/njets2/pretraining/fold1/2026-02-03/0_17-28-29/model_checkpoint.pth'
ckpt_pth_njets2_fold2 = 'Categorizer_results/njets2/pretraining/fold2/2026-02-03/0_17-31-14/model_checkpoint.pth'

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
    weights: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    class_weights: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    process: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None



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

def equi_populated_bins(data, n_bins):

    data = np.asarray(data)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(data, quantiles)
    return bin_edges

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


# ----- plotting -----

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


# ----- main -----

def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model")

    args = Args().parse_args()

    if args.njets == 'inclusive':
        ckpt_pth_fold1 = ckpt_pth_inclusive_fold1
        ckpt_pth_fold2 = ckpt_pth_inclusive_fold2
    elif args.njets == 'njets0':
        ckpt_pth_fold1 = ckpt_pth_njets0_fold1
        ckpt_pth_fold2 = ckpt_pth_njets0_fold2
    elif args.njets == 'njets1':
        ckpt_pth_fold1 = ckpt_pth_njets1_fold1
        ckpt_pth_fold2 = ckpt_pth_njets1_fold2
    elif args.njets == 'njets2':
        ckpt_pth_fold1 = ckpt_pth_njets2_fold1
        ckpt_pth_fold2 = ckpt_pth_njets2_fold2

    model1 = load_model(INPUT_DIM, ckpt_pth_fold1, device)
    model2 = load_model(INPUT_DIM, ckpt_pth_fold2, device)

    logger.info("Loading data")

    # --- fold1 --- 

    train1 = torch.load(f'../data/{args.njets}/train1.pt', weights_only=False)
    train2 = torch.load(f'../data/{args.njets}/train2.pt', weights_only=False)
    val1 = torch.load(f'../data/{args.njets}/val1.pt', weights_only=False)
    val2 = torch.load(f'../data/{args.njets}/val2.pt', weights_only=False)
    
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
                                                              
    probs_nFFt1 = predict_probabilities(model1, train2.X.ss[train2.process.ss > 1], device)
    probs_nFFv1 = predict_probabilities(model1, val2.X.ss[val2.process.ss > 1], device)
    probs_nFFt2 = predict_probabilities(model2, train1.X.ss[train1.process.ss > 1], device)
    probs_nFFv2 = predict_probabilities(model2, val1.X.ss[val1.process.ss > 1], device)

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

    weights_Wjets = torch.concat(
        [train2.weights.ss[train2.process.ss == 1], val2.weights.ss[val2.process.ss == 1], 
         train1.weights.ss[train1.process.ss == 1], val1.weights.ss[val1.process.ss == 1]], 
         dim = 0).detach().cpu().numpy()
    weights_data = torch.concat(
        [train2.weights.ss[train2.process.ss == 0], val2.weights.ss[val2.process.ss == 0],
         train1.weights.ss[train1.process.ss == 0], val1.weights.ss[val1.process.ss == 0]], 
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
    weights_tt_bar_J = torch.concat(
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
        [train2.weights.ss[train2.process.ss > 1], 
             val2.weights.ss[val2.process.ss > 1],
         train1.weights.ss[train1.process.ss > 1], 
             val1.weights.ss[val1.process.ss > 1]], 
         dim = 0).detach().cpu().numpy()

    logger.info(" ------- Plotting NN outputs ------- ")

    probs = [probs_Wjets, probs_embedding, probs_diboson_J, probs_diboson_L, probs_DYjets_J, probs_DYjets_L, probs_ST_J, probs_ST_L, probs_ttbar_J, probs_ttbar_L]
    weights = [weights_Wjets, weights_embedding, weights_diboson_J, weights_diboson_L, weights_DYjets_J, weights_DYjets_L, weights_ST_J, weights_ST_L, weights_tt_bar_J, weights_ttbar_L] 
    labels = ['Wjets', 'embedding', 'diboson_J', 'diboson_L', 'DYjets_J', 'DYjets_L', 'ST_J', 'ST_L', 'ttbar_J', 'ttbar_L']
    colors = ['#e76300', '#ffa90e', '#9f887e', '#94a4a2', '#b9ac70', '#3f90da', '#717581', '#5882ae', '#964c88' ,'#615fc8' ]

    probs_Wjets_list = [probs_Wjetst1, probs_Wjetsv1, probs_Wjetst2, probs_Wjetsv2]
    weights_Wjets_list = [train2.weights.ss[train2.process.ss == 1].detach().cpu().numpy(), val2.weights.ss[val2.process.ss == 1].detach().cpu().numpy(), train1.weights.ss[train1.process.ss == 1].detach().cpu().numpy(), val1.weights.ss[val1.process.ss == 1].detach().cpu().numpy() ]
    probs_nFF_list = [probs_nFFt1, probs_nFFv1, probs_nFFt2, probs_nFFv2]
    weights_nFF_list = [train2.weights.ss[train2.process.ss > 1].detach().cpu().numpy(), val2.weights.ss[val2.process.ss > 1].detach().cpu().numpy(), train1.weights.ss[train1.process.ss > 1].detach().cpu().numpy(), val1.weights.ss[val1.process.ss > 1].detach().cpu().numpy()]
    probs_data_list = [probs_datat1, probs_datav1, probs_datat2, probs_datav2]

    bins_qcd = np.linspace(0, 1, 21)


    weights_qcd_list = []
    for (
        probs_Wjets_i, weights_Wjets_i,
        probs_nFF_i,   weights_nFF_i,
        probs_data_i
    ) in zip(
        probs_Wjets_list, weights_Wjets_list,
        probs_nFF_list,   weights_nFF_list,
        probs_data_list
    ):
        
        bins = equi_populated_bins(probs_data_i, 20)

        # Histograms
        hist_ff, edges = np.histogram(
            probs_Wjets_i, bins=bins_qcd, weights=weights_Wjets_i
        )
        hist_nff, _ = np.histogram(
            probs_nFF_i, bins=bins_qcd, weights=weights_nFF_i
        )
        hist_data, _ = np.histogram(
            probs_data_i, bins=bins_qcd
        )

        # Bin-wise QCD weights
        binwise_QCD_weights = hist_data - hist_nff - hist_ff
        binwise_QCD_weights = np.clip(binwise_QCD_weights, 0, None)

        # Safe normalization
        mask = hist_data > 0
        binwise_QCD_weights[mask] /= hist_data[mask]
        binwise_QCD_weights[~mask] = 0.0

        # Assign per-event weights
        indices = np.digitize(probs_data_i, edges) - 1
        indices = np.clip(indices, 0, len(bins_qcd) - 2)

        weights_qcd_i = binwise_QCD_weights[indices]
        weights_qcd_list.append(weights_qcd_i)

    weights_qcd_all = np.concatenate(weights_qcd_list, axis=0)

    logger.info(f'len loaded data X train1: {len(train1.X.ss[train1.process.ss == 0])}')
    logger.info(f'len of the according weights: {len(weights_qcd_list[2])}')

    logger.info(f'len loaded data X val1: {len(val1.X.ss[val1.process.ss == 0])}')
    logger.info(f'len of the according weights: {len(weights_qcd_list[3])}')

    np.save(f'../data/{args.njets}/qcd_weights_train2.npy', weights_qcd_list[0])
    np.save(f'../data/{args.njets}/qcd_weights_val2.npy', weights_qcd_list[1])
    np.save(f'../data/{args.njets}/qcd_weights_train1.npy', weights_qcd_list[2])
    np.save(f'../data/{args.njets}/qcd_weights_val1.npy', weights_qcd_list[3])




    # --- Style settings ---
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

    # --- Binning ---
    bins = equi_populated_bins(probs_data, 20)

    # --- Data histogram ---
    data_counts, bin_edges = np.histogram(probs_data, bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Errors
    y_error = np.sqrt(data_counts)
    x_error = 0.5 * np.diff(bins)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(15, 9))

    # --- Stacked simulation ---
    ax.hist(
        probs,
        bins=bins,
        weights=weights,
        histtype='barstacked',
        label=labels,
        color=colors
    )

    # --- Data points ---
    ax.errorbar(
        bin_centers,
        data_counts,
        yerr=y_error,
        xerr=x_error,
        fmt='o',
        color='black',
        label='Data',
        markersize=5
    )

    # --- Axis styling ---
    ax.set_xlabel("NN output")
    ax.set_ylabel("Events")

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax.set_ylim(0, 1.2*np.max(data_counts))
    ax.legend(loc='upper center', ncol=4, frameon=False)

    ax.tick_params(direction='in', top=True, right=True)

    # --- Optional CMS text ---
    # ax.text(0.02, 0.92, "CMS Simulation", transform=ax.transAxes,
    #         fontsize=18, fontweight='bold', va='top')
    # ax.text(0.02, 0.85, "L = 138 fb$^{-1}$, 13 TeV", transform=ax.transAxes,
    #         fontsize=14, va='top')

    # --- Save ---
    fig.tight_layout()
    fig.savefig(f'../plots/{args.njets}/results_pretraining_equi.png')
    fig.savefig(f'../plots/{args.njets}/results_pretraining_equi.pdf')

    plt.close(fig)

    plt.figure()
    plt.hist(probs_embedding, bins = bins, label = 'embedding', color = '#ffa90e')
    plt.savefig('delme.png')
# ---------------

if __name__ == "__main__":
    main()