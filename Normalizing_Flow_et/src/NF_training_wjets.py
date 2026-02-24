import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
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



SEED = 42

t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

t.set_num_threads(8)


variables = [
    'pt_1',
    'pt_2',
    'm_vis',
    'deltaR_ditaupair',
    'pt_vis',
    'eta_1',
    'eta_2',
]

dim = len(variables)

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

N_EPOCHS_MAX = 500
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

# ------ model ------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        last = in_dim

        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h

        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Affine Coupling Layer
# -------------------------
class AffineCoupling(nn.Module):
    def __init__(self, dim, mask, hidden_dims=(128, 128), s_scale=2.0):
        super().__init__()

        self.dim = dim
        self.register_buffer("mask", mask)

        self.st_net = MLP(in_dim=dim, out_dim=2 * dim, hidden_dims=hidden_dims)
        self.s_scale = s_scale

    def forward(self, x):
        x_masked = x * self.mask

        s, t = torch.chunk(self.st_net(x_masked), 2, dim=-1)
        s = torch.tanh(s) * self.s_scale

        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = ((1 - self.mask) * s).sum(dim=-1)

        return y, log_det

    def inverse(self, y):
        y_masked = y * self.mask

        s, t = torch.chunk(self.st_net(y_masked), 2, dim=-1)
        s = torch.tanh(s) * self.s_scale

        x = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        return x



class RealNVP(nn.Module):
    """
    Stack of affine coupling layers with alternating masks.
    Base distribution: standard Normal.
    """
    def __init__(self, dim, n_layers=6, hidden_dims=(128, 128), s_scale=2.0, device= torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        torch.device(device)
        self.dim = dim
        logger.info(f"Dimension of RealNVP input: {self.dim}")
        base_mask = torch.tensor([i % 2 for i in range(dim)], dtype=torch.float32)

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

    def log_prob(self, x):
        """Compute log p(x) via change of variables."""
        z, log_det = self.f(x)
        # base log prob (diagonal Normal)
        std = torch.exp(self.base_log_std)
        log_pz = (-0.5 * (((z - self.base_mean) / std) ** 2).sum(dim=-1)
                  - 0.5 * self.dim * math.log(2 * math.pi)
                  - self.base_log_std.sum())
        return log_pz + log_det

    def sample(self, n):
        """Sample x by drawing z from base and mapping through inverse."""
        std = torch.exp(self.base_log_std)
        z = self.base_mean + std * torch.randn(n, self.dim, device=self.base_mean.device)
        x_scaled = self.f_inv(z)
        # map back to original (un-scaled) space: x = x_scaled * scale + shift
        x = x_scaled * self._scaler_scale.to(x_scaled.device) + self._scaler_shift.to(x_scaled.device)
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
        # Both shift and scale must be provided (no partial initialization)
        if shift is None or scale is None:
            msg = (
                "shift and scale must be both provided (not None)."
                " To reset to defaults explicitly set shift=0 and scale=1."
            )
            logger.error(msg)
            raise ValueError(msg)

        # convert numpy arrays to tensors if needed
        shift = torch.from_numpy(shift) if isinstance(shift, np.ndarray) else shift
        scale = torch.from_numpy(scale) if isinstance(scale, np.ndarray) else scale

        if self._is_initialized:
            logger.warning("Scaler already initialized. Overwriting the current values.")

        self._scaler_shift.data[:] = shift
        self._scaler_scale.data[:] = scale

    def apply_scaler(self, x):
        return (x - self._scaler_shift.to(x.device)) / self._scaler_scale.to(x.device)



    def forward(self, X):
        Xs = self.apply_scaler(X)
        # log prob of the scaled input
        logp_scaled = self.log_prob(Xs)
        # account for the scaling Jacobian: Xs = (X - shift)/scale -> det = prod(1/scale)
        # so log|det| = -sum(log(scale)). Add this constant per-sample.
        log_det_scale = -torch.log(self._scaler_scale.to(X.device)).sum()
        return logp_scaled + log_det_scale

# ------ functions and masks --------

def mask_DR(df):

    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 > 70)
    mask_DR = (mask_a1 & mask_a2  & mask_a4 & mask_a5 & mask_a6)

    return df[mask_DR].copy()

def mask_preselection_loose(df):
    mask_eta = (df.eta_1 <= 2.1) & (df.eta_2 <= 2.3)
    mask_pt = (df.pt_1 >= 33) & (df.pt_2 >= 30)
    mask_tau_decay_mode = (df.tau_decaymode_2 == 0) | (df.tau_decaymode_2 == 1) | (df.tau_decaymode_2 == 10) | (df.tau_decaymode_2 == 11)
    return df[mask_eta & mask_pt & mask_tau_decay_mode]


def SR_like(df):
    mask_s1 = (df.id_tau_vsJet_Tight_2 > 0.5)
    return(df[mask_s1])

# AR-like mask
def AR_like(df):
    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5) & (df.id_tau_vsJet_Tight_2 < 0.5))

    return(df[mask_a1])


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
    _df = df

    return _component_collection(
        X=_df[training_var].to_numpy(dtype=np.float32),
        weights=_df["weight_wjets"].to_numpy(dtype=np.float32),

    )

@dataclass
class _same_sign_opposite_sign_split(metaclass=helper.CollectionMeta):
    ss: Union[torch.Tensor, pd.DataFrame, np.ndarray]
    os: Union[torch.Tensor, pd.DataFrame, np.ndarray]


@dataclass
class _component_collection(metaclass=helper.CollectionMeta):
    _: KW_ONLY
    X: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None
    weights: Union[torch.Tensor, pd.DataFrame, np.ndarray, None] = None




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

    data_complete = pd.read_feather('../../data/data_complete.feather')



    data_DR = mask_DR(data_complete)

    data_DR = data_DR[(data_DR.process == 0) & (data_DR.OS == True)].reset_index(drop=True)

    train1, val1, train2, val2 = split_even_odd(data_DR)

    train1_AR_like = mask_preselection_loose(AR_like(train1))
    val1_AR_like = mask_preselection_loose(AR_like(val1))
    train1_SR_like = mask_preselection_loose(SR_like(train1))
    val1_SR_like = mask_preselection_loose(SR_like(val1))    

    weight_corr_factor = (
        pd.concat([train1_AR_like['weight_wjets'], val1_AR_like['weight_wjets']]).sum()
        /
        pd.concat([train1_SR_like['weight_wjets'], val1_SR_like['weight_wjets']]).sum()
    )

    if args.region == "AR-like":
        train1 = train1_AR_like
        val1 = val1_AR_like
    elif args.region == "SR-like":
        train1 = train1_SR_like
        val1 = val1_SR_like

    if args.njets == "njet0":
        train1 = train1[train1.njet == 0]
        val1 = val1[val1.njet == 0]
    elif args.njets == "njet1":
        train1 = train1[train1.njet == 1]
        val1 = val1[val1.njet == 1]
    elif args.njets == "njet2":
        train1 = train1[train1.njet >= 2]
        val1 = val1[val1.njet >= 2]


    print(data_DR['weight_wjets'])

    train1 = get_my_data(train1, variables).to_torch(device=None)
    val1 = get_my_data(val1, variables).to_torch(device=None)


    X_train = train1.X
    X_val = val1.X
    weights_train = train1.weights
    weights_val = val1.weights  

    weights_train = weights_train / torch.sum(weights_train)
    weights_val   = weights_val / torch.sum(weights_val)


    # Normalize weights
    if args.region == "SR-like":
        weights_train = weights_train * weight_corr_factor
        weights_val = weights_val * weight_corr_factor
    elif args.region == "AR-like":
        pass  # no correction needed for AR-like region

    # -------------------------------------
    #  DataLoaders
    # -------------------------------------
    train_loader = DataLoader(
        TensorDataset(X_train, weights_train),
        batch_size=config.bsize_train,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    val_loader = DataLoader(
        TensorDataset(X_val, weights_val),
        batch_size=config.bsize_val,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    # --- model setup ---

    dim  = len(variables)

    shift = X_train.mean(dim=0)
    scale  = X_train.std(dim=0, unbiased=False).clamp_min(1e-12) 

    model = RealNVP(dim=dim, n_layers=config.n_layers, hidden_dims=(config.hidden_dims,), s_scale=config.s_scale).to(device)
    model.initialize_scaler(shift=shift, scale=scale)           
    #optimizer = torch.optim.NAdam(model.parameters(), lr=config.lr)
    optimizer= torch.optim.AdamW(model.parameters(), lr=config.lr)
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

        for Xb, Wb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            Wb = Wb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=False):

                log_px = model(Xb)
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
            for Xb, Wb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                Wb = Wb.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=False):
                    log_px = model(Xb)
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
    paths_training = StorePathHelper(directory=f"Training_results_new/{args.njets}/{args.region}")
    paths_plots = StorePathHelper(directory=f"NF_results_new/{args.njets}/{args.region}")

    torch.save(checkpoint, paths_training.autopath.joinpath("model_checkpoint.pth"))

    pd.DataFrame(log_rows).to_pickle(str(paths_training.autopath.joinpath('training_logs.pkl')))


    with open(paths_training.autopath.joinpath("config.yaml"), "w") as f:
        yaml.dump(config, f)

    logger.info("Model saved successfully")

# --------------

if __name__ == "__main__":
    main()