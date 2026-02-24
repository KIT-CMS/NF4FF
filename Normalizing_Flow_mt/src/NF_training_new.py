import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from classes.config_loader import load_config
from classes.dataloading import Datasets
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
import itertools as itt
import uproot
import logging
import pathlib
import pickle
import random
from collections import defaultdict
from dataclasses import KW_ONLY, dataclass, is_dataclass
from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Protocol, Tuple,
                    Union, runtime_checkable)
import CODE.HELPER as helper
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch as t
import torch.nn as nn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, scale
from tqdm import tqdm
from contextlib import contextmanager
from typing import (Any, Callable, Dict, Generator, Iterable, Iterator, List,
                    Optional, Tuple, Type, Union, get_args, get_origin)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
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
PATIENCE = 20
#N_SAMPLES = 1000000../src/Training_results/AR-like/loose/2026-01-19/0_16-45-26/model_checkpoint.pth


# ----- TAP Arguments -----
class Args(Tap):
    preselection: Literal["loose", "tight"] = "loose"
    region: Literal["AR-like", "SR-like"] = "AR-like"
    njets: Literal["all", "0", "1", "2"] = "all"

# ----- Configuration -----
from dataclasses import dataclass


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

@dataclass
class _component_collection(metaclass=helper.CollectionMeta):
    _: KW_ONLY
    X: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    Njets: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    weights: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None


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

class MLP(nn.Module):
    """Simple MLP used to parameterize s(x) and t(x)."""
    def __init__(self, in_dim, out_dim, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)   #creates a container and, when called, passes the input through each layer in order
    def forward(self, x):
        return self.net(x)
        
class AffineCoupling(nn.Module):

    def __init__(self, dim, mask, hidden_dims=(128, 128), s_scale=2.0,):
        super().__init__()
        self.dim = dim
        # mask is shape (dim,) with entries 0 or 1
        self.register_buffer('mask', mask)          #mask is passed to the model, but not trained
        # network outputs both s and t (concatenate)
        self.st_net = MLP(in_dim=dim, out_dim=2*dim, hidden_dims=hidden_dims)
        self.s_scale = s_scale  # scale s via tanh to stabilize exp(s)


    def reset_parameters(self):
        # Initialize hidden layers for ReLU
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                is_last = (i == len(self.net) - 1)
                if is_last:
                    # ZERO init for s,t head to start near identity
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.zeros_(layer.bias)


    def forward(self, x):
        """
        Forward transform: x -> y, returns y and logdet.
        """
        x_masked = x * self.mask                                    # creates the half masked dataset
        st = self.st_net(x_masked)                                  # masked vector is passed through the nn here 
        s, t = torch.chunk(st, chunks=2, dim=-1)                    
        # stabilize s using tanh                                    
        s = torch.tanh(s) * self.s_scale
        # transform only the (1 - mask) part
        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)     #
        # log det = sum of s over transformed dims
        log_det = ((1 - self.mask) * s).sum(dim=-1)
        return y, log_det

    def inverse(self, y):
        """
        Inverse transform: y -> x (needed for sampling).
        """
        y_masked = y * self.mask
        st = self.st_net(y_masked)
        s, t = torch.chunk(st, chunks=2, dim=-1)
        s = torch.tanh(s) * self.s_scale
        # invert affine transform on (1 - mask) part
        x = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        return x

class RealNVP_with_njets(nn.Module):
    """
    Stack of affine coupling layers with alternating masks.
    Base distribution: standard Normal.
    """
    def __init__(self, dim, n_layers=6, hidden_dims=(128, 128), s_scale=2.0, device= torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        torch.device(device)
        self.dim = dim
        logger.info(f"Dimension of RealNVP input: {self.dim}")
        base_mask = torch.randint(0, 2, (dim,), dtype=torch.float32)

        masks = []
        for i in range(n_layers):
            if i % 2 == 0:
                masks.append(base_mask)
            else:
                masks.append(1 - base_mask)

        # create list of layers, 
        self.couplings = nn.ModuleList([
            AffineCoupling(dim=dim, mask=m, hidden_dims=hidden_dims, s_scale=s_scale)
            for m in masks
        ])

        # Learnable base distribution parameters (optional; here fixed to standard normal)
        self.register_buffer('base_mean', torch.zeros(dim))
        self.register_buffer('base_log_std', torch.zeros(dim))
                # StandardScaler or RobustScaler
        self.register_buffer("_scaler_shift", torch.full((dim,), 0.0))
        self.register_buffer("_scaler_scale", torch.full((dim,), 1.0))
        self.njets_head = nn.Sequential(
            nn.Linear(dim, 64),   # or dim, or some hidden feature size you like
            nn.ReLU(),
            nn.Linear(64, 3)      # logits for classes: 0, 1, >=2
        )


    def _init_permute(self, m):
        # For invertible mixing layers, orthogonal init is robust
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
        elif isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_fn(self, module):
        # Generic global init (use cautiously; don't override the s,t zero head)
        if isinstance(module, nn.Linear) and module not in [self.permute]:
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)


    def f(self, x):
        """Forward through all couplings: x -> z. Returns z and sum log-dets."""
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.couplings:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def f_inv(self, z):
        """Inverse through all couplings: z -> x."""
        x = z
        # inverse in reverse order
        for layer in reversed(self.couplings):
            x = layer.inverse(x)
        return x



    def log_prob(self, x_cont, njets_class):
        """
        x_cont: (B, D) continuous variables
        njets_class: (B,) LongTensor in {0,1,2}
        """

        # ---- continuous flow part ----
        z, log_det = self.f(x_cont)
        std = torch.exp(self.base_log_std)
        log_pz = (-0.5 * (((z - self.base_mean) / std) ** 2).sum(dim=-1)
                - 0.5 * self.dim * math.log(2 * math.pi)
                - self.base_log_std.sum())
        log_flow = log_pz + log_det

        # ---- categorical njets part ----
        log_njets = self.log_prob_njets(x_cont, njets_class)

        # ---- total log-likelihood ----
        return log_flow + log_njets





    def log_prob_njets(self, x_cont, njets_class):
        """
        x_cont: (B, D) continuous features
        njets_class: (B,) LongTensor with values {0,1,2}
        """
        logits = self.njets_head(x_cont)   # (B, 3)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, 3)

        # pick the log prob of the correct class
        return log_probs[torch.arange(len(njets_class)), njets_class]


    def sample(self, n):
        """Sample x by drawing z from base and mapping through inverse."""
        std = torch.exp(self.base_log_std)
        z = self.base_mean + std * torch.randn(n, self.dim, device=self.base_mean.device)
        x = self.f_inv(z)
        return x
    

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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(self.apply_scaler(X))
    
def map_njets_to_3cls(njets_raw: torch.Tensor) -> torch.Tensor:
    # Map: 0 -> 0, 1 -> 1, >=2 -> 2
    # Keep device & dtype aligned with input
    cls = torch.empty_like(njets_raw, dtype=torch.long, device=njets_raw.device)
    cls[njets_raw == 0] = 0
    cls[njets_raw == 1] = 1
    cls[njets_raw >= 2] = 2
    return cls

def sample_from_flow(model: RealNVP_with_njets, n_samples:int, device:torch.device) -> np.ndarray:
    model = model
    with rng_seed(42):
        z = torch.randn(n_samples, model.dim, device=device)  # latent samples
        with torch.no_grad():
            x_samples = model.f_inv(z)  # shape: [n_samples, dim]

        x_samples = x_samples.cpu().numpy()
    return x_samples

def get_my_data(df, training_var):
    _df = df

    return _component_collection(
        X=_df[training_var].to_numpy(dtype=np.float32),
        Njets=_df["njets"].to_numpy(dtype=np.float32),
        weights=_df["weight"].to_numpy(dtype=np.float32),
        # class_weights=_df["class_weights"].to_numpy(),
        # process=_df["process"].to_numpy(dtype=np.float32),
    )

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
            label="MC data",
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

    # -------------------------------------
    #  RNG setup
    # -------------------------------------
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # -------------------------------------
    #  Argument parsing
    # -------------------------------------
    args = Args().parse_args()
    config_path = '../configs/config_NF.yaml'

    raw = load_config(config_path)
    config = Config.from_dict(raw)

    # -------------------------------------
    #  Device
    # -------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # -------------------------------------
    #  Load physics dataset
    # -------------------------------------
    dataset = Datasets()
    preselection = args.preselection
    region = args.region

    dataset_map = {
        ('tight','AR-like'): dataset.AR_like_tight,
        ('loose','AR-like'): dataset.AR_like_loose,
        ('tight','SR-like'): dataset.SR_like_tight,
        ('loose','SR-like'): dataset.SR_like_loose
    }

    data = dataset_map.get((preselection, region))
    if data is None:
        raise ValueError(f"Invalid combination preselection={preselection}, region={region}")

    # extract data columns
    data_pt = get_my_data(data, ['pt_1', 'pt_2', 'm_vis', 'deltaR']).to_torch(device=None)

    X = data_pt.X
    Njets = data_pt.Njets
    weights = data_pt.weights

    # -------------------------------------
    #  Train/Val split
    # -------------------------------------
    X_train, X_val, Njets_train, Njets_val, weights_train, weights_val = train_test_split(
        X, Njets, weights, test_size=0.2, random_state=42
    )

    # Normalize weights
    weights_train = weights_train / torch.sum(weights_train)
    weights_val   = weights_val / torch.sum(weights_val)

    # -------------------------------------
    #  DataLoaders
    # -------------------------------------
    train_loader = DataLoader(
        TensorDataset(X_train, Njets_train, weights_train),
        batch_size=config.bsize_train,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    val_loader = DataLoader(
        TensorDataset(X_val, Njets_val, weights_val),
        batch_size=config.bsize_val,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    # -------------------------------------
    #  Model
    # -------------------------------------
    dim = 4
    shift = X_train.mean(dim=0)
    scale = X_train.std(dim=0, unbiased=False).clamp_min(1e-12)

    model = RealNVP_with_njets(
        dim=dim,
        n_layers=config.n_layers,
        hidden_dims=(config.hidden_dims,),
        s_scale=config.s_scale,
        device=device
    ).to(device)

    # scaler initialization
    model.initialize_scaler(shift=shift, scale=scale)

    # optimizer + scheduler
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

    # AMP
    scaler = torch.amp.GradScaler('cuda',enabled=config.use_amp)

    # -------------------------------------
    #  Training bookkeeping
    # -------------------------------------
    NLL_training = []
    NLL_validation = []
    best_val_nll = float('inf')
    counter = 0
    log_rows = []

    # -------------------------------------
    #  Training Loop
    # -------------------------------------
    for epoch in range(1, config.n_epochs + 1):
        epoch_start = time.time()

        # --------------------
        #  TRAIN
        # --------------------
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0

        for Xb, Nb, Wb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            Nb = Nb.to(device, non_blocking=True)
            Wb = Wb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=config.use_amp):
                nj_cls = map_njets_to_3cls(Nb)
                log_px = model.log_prob(Xb, nj_cls)
                loss = (-(log_px) * Wb).sum() / Wb.sum()

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * Wb.sum().item()
            train_weight_sum += Wb.sum().item()

        avg_train_nll = train_loss_sum / train_weight_sum
        NLL_training.append(avg_train_nll)

        # --------------------
        #  VALIDATION
        # --------------------
        model.eval()
        val_loss_sum = 0.0
        val_weight_sum = 0.0

        with torch.no_grad():
            for Xb, Nb, Wb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                Nb = Nb.to(device, non_blocking=True)
                Wb = Wb.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=config.use_amp):
                    nj_cls = map_njets_to_3cls(Nb)
                    log_px = model.log_prob(Xb, nj_cls)
                    vloss = (-(log_px) * Wb).sum() / Wb.sum()

                val_loss_sum += vloss.item() * Wb.sum().item()
                val_weight_sum += Wb.sum().item()

        avg_val_nll = val_loss_sum / val_weight_sum
        NLL_validation.append(avg_val_nll)

        scheduler.step(avg_val_nll)
        epoch_time = time.time() - epoch_start

        log_rows.append({
        "epoch": epoch,
        "train_loss": avg_train_nll,
        "val_loss": avg_val_nll,
        "lr": scheduler.get_last_lr(),
        "time_s": epoch_time,
        "type": "epoch",
         })


        # --------------------
        #  Early stopping
        # --------------------
        if avg_val_nll < best_val_nll:
            best_val_nll = avg_val_nll
            counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
        else:
            counter += 1

        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: train={avg_train_nll:.6f}, val={avg_val_nll:.6f}, lr = {scheduler.get_last_lr()}")

        if counter >= PATIENCE:
            logger.info("Early stopping triggered.")
            break


    # -------------------------------------
    #  Save training artifacts
    # -------------------------------------
    paths_training = StorePathHelper(directory=f"Training_results_new/{args.njets}/{args.region}/{args.preselection}")
    paths_plots = StorePathHelper(directory=f"NF_results_new/{args.njets}/{args.region}/{args.preselection}")

    torch.save(checkpoint, paths_training.autopath.joinpath("model_checkpoint.pth"))
    np.savetxt(paths_training.autopath.joinpath("NLL_training.txt"), np.array(NLL_training))
    np.savetxt(paths_training.autopath.joinpath("NLL_val.txt"), np.array(NLL_validation))

    pd.DataFrame(log_rows).to_pickle(str(paths_training.autopath.joinpath('training_logs.pkl')))


    with open(paths_training.autopath.joinpath("config.yaml"), "w") as f:
        yaml.dump(config, f)

    logger.info("Model saved successfully")

    # -------------------------------------
    #  Sampling + Plots
    # -------------------------------------
    n_samples = 1_000_000
    x_samples = sample_from_flow(model, n_samples, device)

    train_np = torch.cat([X_train, X_val], dim=0).detach().cpu().numpy()
    data_weights = torch.cat([weights_train, weights_val], dim=0).detach().cpu().numpy()

    flow = x_samples.T
    data = train_np.T

    data_min = [20, 20, 0, 0]
    data_max = [80, 80, 225, 4]
    bin_width = [1, 1, 2, 0.1]
    DR_min = [30, 30, 0, 0]
    DR_max = [80, 80, 225, 4]

    title = ['pt_1', 'pt_2', 'm_vis', 'deltaR']
    xlabels = [
        r'$p^{\mu}_T$ (GeV)',
        r'$p^{\tau_h}_T$ (GeV)',
        r'$m_{vis}$ (GeV)',
        r'$\Delta R$'
    ]

    plot_hist_ratio(
        flow=flow,
        data=data,
        data_weights=data_weights,
        data_min=data_min,
        data_max=data_max,
        DR_min=DR_min,
        DR_max=DR_max,
        bin_width=bin_width,
        xlabels=xlabels,
        title=title,
        paths=paths_plots
    )
# --------------

if __name__ == "__main__":
    main()