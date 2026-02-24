import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from classes.models import RealNVP
from classes.config_loader import load_config
from classes.training import train_epoch, val_epoch
from classes.path_managment import StorePathHelper
import logging
from CustomLogging import setup_logging
import yaml
import numpy as np
import random
from contextlib import contextmanager
from tap import Tap
from typing import Literal, Generator
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Any, Dict
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch as t
import CODE.HELPER as helper
import logging
import random
from dataclasses import KW_ONLY, dataclass
from typing import (Any, Callable, Dict, Iterable, List, Protocol, Tuple,
                    Union, runtime_checkable)
from contextlib import contextmanager
from typing import (Any, Callable, Dict, Generator, Iterable, Iterator, List,
                    Optional, Tuple, Type, Union, get_args, get_origin)
from sklearn.model_selection import train_test_split

from CustomLogging import setup_logging
import time


# ------ logger -----

logger = setup_logging(logger=logging.getLogger(__name__))

# ------ RNG handling ------------------

@contextmanager
def rng_seed(seed: int) -> Generator[None, None, None]:
    np_state, py_state = np.random.get_state(), random.getstate()
    torch_state = torch.get_rng_state()
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.set_rng_state(torch_state)

# ----- constants -----

N_EPOCHS_MAX = 2_000
PATIENCE = 30
N_SAMPLES = 1000000


# ----- TAP Arguments -----
class Args(Tap):
    region: Literal["AR-like", "SR-like"] = "AR-like"
    njets: Literal["all", "0", "1", "2"] = "all"


@dataclass
class _component_collection(metaclass=helper.CollectionMeta):
    _: KW_ONLY
    X: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    Njets: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    weights: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    class_weights: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    process: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None


# ----- Configuration -----


@dataclass
class Config:
    # training
    bsize_train: int
    bsize_val: int
    bsize_test: int
    grad_clip: float
    n_epochs: int
    use_amp: bool
    s_scale_max: float

    # model
    n_layers: int
    hidden_dims: int
    s_scale: float

    # optimizer
    lr: float
    weight_decay: float
    eps: float

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
        """Construct from the original nested YAML structure."""
        training = cfg["training"]
        model = cfg["model"]
        optimizer = cfg["optimizer"]
        scheduler = cfg["scheduler"]

        return Config(
            # training
            bsize_train=training["bsize_train"],
            bsize_val=training["bsize_val"],
            bsize_test=training["bsize_test"],
            grad_clip=training["grad_clip"],
            n_epochs=training["n_epochs"],
            use_amp=training["use_amp"],
            s_scale_max=training["s_scale_max"],

            # model
            n_layers=model["n_layers"],
            hidden_dims=model["hidden_dims"],
            s_scale=model["s_scale"],

            # optimizer
            lr=optimizer["lr"],
            weight_decay=optimizer["weight_decay"],
            eps=optimizer["eps"],

            # scheduler
            scheduler_step_size=scheduler["step_size"],
            scheduler_gamma=scheduler["gamma"],
            scheduler_factor=scheduler["factor"],
            scheduler_patience=scheduler["patience"],
            scheduler_threshold=scheduler["threshold"],
            scheduler_cooldown=scheduler["cooldown"],
            scheduler_min_lr=scheduler["min_lr"],
            scheduler_eps=scheduler["eps"],
        )

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----- functions -----

def create_4D_numpy_data(events) -> np.ndarray:
    return np.vstack([events.pt_1, events.pt_2, events.m_vis, events.deltaR]).T

def split_array_80_20_random(arr):
    arr_copy = arr.copy()
    random.shuffle(arr_copy)
    split_index = int(len(arr_copy) * 0.8)
    return arr_copy[:split_index], arr_copy[split_index:]

def sample_from_flow(model: RealNVP, n_samples:int, device:torch.device) -> np.ndarray:
    model = model
    with rng_seed(42):
        z = torch.randn(n_samples, model.dim, device=device)  # latent samples
        with torch.no_grad():
            x_samples = model.f_inv(z)  # shape: [n_samples, dim]

        x_samples = x_samples.cpu().numpy()
    return x_samples

def train_full_batch(model, X, weights, optimizer, device, grad_clip):
    model.train()

    X = X.to(device)
    weights = weights.to(device)

    log_px = model.log_prob(X)
    loss = (-log_px * weights).sum()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    if grad_clip is not None:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    # keep the same semantic meaning as before
    avg_nll = loss.item()

    return avg_nll

@torch.no_grad()
def val_full_batch(model, X, weights, device):
    model.eval()

    X = X.to(device)
    weights = weights.to(device)

    log_px = model.log_prob(X)
    loss = (-log_px * weights).sum()

    # same meaning as before: average over number of batches (=1)
    avg_nll = loss.item()

    return avg_nll

def plot_hist_ratio(
    flow,
    data,
    data_weights,
    data_min,
    data_max,
    DR_min,
    DR_max,
    bin_width,
    xlabels,
    title,
    paths
):
    for i in range(len(flow)):

        # ------ cutting data -----

        mask_min_flow = flow[i] > data_min[i]
        mask_max_flow = flow[i] < data_max[i]
        mask_min_data = data[i] > data_min[i]
        mask_max_data = data[i] < data_max[i]

        flow_masked = flow[i][mask_min_flow & mask_max_flow]
        data_masked = data[i][mask_min_data & mask_max_data]
        data_weights_masked = data_weights[mask_min_data & mask_max_data]

        # ------ calculating ratio normalization ------

        flow_DR = flow[i][(flow[i] > DR_min[i]) & (flow[i] < DR_max[i])]
        data_DR = data[i][(data[i] > DR_min[i]) & (data[i] < DR_max[i])]
        data_weights_DR = data_weights[(data[i] > DR_min[i]) & (data[i] < DR_max[i])]

        ratio_flow_data = len(flow_DR) / len(data_DR)

        hist_bins = np.arange(
            data_min[i],
            data_max[i] + bin_width[i],
            bin_width[i]
        )

        flow_weights = (
            np.ones(len(flow_masked))
            / ratio_flow_data
            * np.mean(data_weights_DR)
        )

        # ------ histograms ------

        counts_flow, _ = np.histogram(
            flow_masked,
            bins=hist_bins,
            weights=flow_weights
        )
        counts_data, _ = np.histogram(
            data_masked,
            bins=hist_bins,
            weights=data_weights_masked
        )

        ratio = np.divide(
            counts_flow,
            counts_data,
            out=np.zeros_like(counts_flow, dtype=float),
            where=counts_data != 0
        )

        # ------ plotting ------

        fig, (ax_top, ax_ratio) = plt.subplots(
            2, 1,
            figsize=(7, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
        )

        ax_top.hist(
            flow_masked,
            bins=hist_bins,
            weights=flow_weights,
            histtype="step",
            linewidth=1.5,
            color="grey",
            label="Sampled from flow",
        )

        ax_top.hist(
            data_masked,
            bins=hist_bins,
            weights=data_weights_masked,
            alpha=0.5,
            label="data",
        )

        ax_top.set_ylabel("counts weighted")
        ax_top.set_yscale("log")
        ax_top.legend(loc="lower right")
        ax_top.grid(True, alpha=0.3)

        ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax_ratio.stairs(
            ratio,
            hist_bins,
            linewidth=1.5,
            color="grey"
        )
        ax_ratio.grid(True)
        ax_ratio.set_xlabel(xlabels[i])
        ax_ratio.set_ylabel("ratio")
        ax_ratio.set_ylim(0.5, 1.5)

        plt.savefig(paths.autopath.joinpath(title[i]))
        plt.savefig(paths.autopath.joinpath(f"{title[i]}.pdf"))
        plt.close(fig)

# ----- main -----

def main():

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    args = Args().parse_args()

    config_path = '../configs/config_NF.yaml'


    raw = load_config(config_path)
    config = Config.from_dict(raw)

    # --- load device ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- load data ---

    if args.region == 'AR-like':
        data = torch.load('../data/data_AR_like.pt')
    elif args.region == 'SR-like':
        data = torch.load('../data/data_SR_like.pt')

    X = data.X
    w = data.weights

    # --- data splitting ---

    X_train, X_val, w_train, w_val = train_test_split(
        X,
        w,
        test_size=0.5,
        random_state=42
    )


    # --- model setup ---

    dim  = 4

    shift = X_train.mean(dim=0)
    scale  = X_train.std(dim=0, unbiased=False).clamp_min(1e-12) 

    model = RealNVP(dim=dim, n_layers=config.n_layers, hidden_dims=(config.hidden_dims,), s_scale=config.s_scale).to(device)
    model.initialize_scaler(shift=shift, scale=scale)           
    optimizer = torch.optim.NAdam(model.parameters(), lr=config.lr)
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

    NLL_training = []
    NLL_validation = []
    best_val_nll = float('inf')
    counter = 0
    global_step = 0
    log_rows = []

    for epoch in range(1, N_EPOCHS_MAX + 1):
        epoch_start = time.time()

        # ----- training -----

        avg_train_nll = train_full_batch(
            model,
            X_train,
            w_train,
            optimizer,
            device,
            config.grad_clip,
        )
        NLL_training.append(avg_train_nll)
        
        
        # ------ val -----
        
        avg_val_nll = val_full_batch(
            model,
            X_val,
            w_val,
            device,
        )
        NLL_validation.append(avg_val_nll)
        
        scheduler.step(avg_val_nll)
        epoch_time = time.time() - epoch_start

        if avg_val_nll < best_val_nll:
            best_val_nll = avg_val_nll
            counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),   # all weights and buffers
                'optimizer_state_dict': optimizer.state_dict(),  # optional
            }
        else:
            counter += 1

        # ----- logging -----

        log_rows.append({
        "epoch": epoch,
        "train_loss": avg_train_nll,
        "val_loss": avg_val_nll,
        "lr": scheduler.get_last_lr(),
        "time_s": epoch_time,
        "type": "epoch",
         })

        if epoch%10 == 1:
            logger.info(f"Epoch {epoch}: train={avg_train_nll:.6f}, val={avg_val_nll:.6f}, LR={scheduler.get_last_lr()}")
        
        if counter >= PATIENCE:
            break
        
    # ----- save results -----
    
    paths_training = StorePathHelper(directory=f"Training_results/{args.njets}/{args.region}")
    paths_plots = StorePathHelper(directory=f"NF_results/{args.njets}/{args.region}")

    torch.save(checkpoint, paths_training.autopath.joinpath("model_checkpoint.pth"))
    np.savetxt(paths_training.autopath.joinpath("NLL_training.txt"), np.array(NLL_training))
    np.savetxt(paths_training.autopath.joinpath("NLL_val.txt"), np.array(NLL_validation))

    with open(paths_training.autopath.joinpath("config.yaml"), "w") as f:
        yaml.dump(config, f)

    logger.info("Model saved successfully")

    # ----- sampling and plotting -----


    n_samples = 1000000
    x_samples = sample_from_flow(model, n_samples, device)



    train_np = X_train.detach().cpu().numpy()
    data_weights = w_train.detach().cpu().numpy()


    data_min = [20, 20, 0, 0]
    data_max = [80, 80, 225, 4]
    bin_width = [1, 1, 2, 0.1]
    title = ['pt_1', 'pt_2', 'm_vis', 'deltaR']
    xlabels = [r'$p^{\mu}_T$ (GeV)', r'$p^{\tau_h}_T$ (GeV)', r'$m_{vis}$ (GeV)', r'$\Delta R$']

    flow = x_samples.T
    data = train_np.T

    DR_min = [30, 30, 0, 0]
    DR_max = [80, 80, 225, 4]

    plot_hist_ratio(
        flow = flow,
        data = data,
        data_weights = data_weights,
        data_min = data_min,
        data_max = data_max,
        DR_min=DR_min,
        DR_max=DR_max,
        bin_width=bin_width,
        xlabels=xlabels,
        title=title,
        paths = paths_plots
    )

# --------------

if __name__ == "__main__":
    main()