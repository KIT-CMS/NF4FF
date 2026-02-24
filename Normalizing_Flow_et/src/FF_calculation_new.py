from __future__ import annotations
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tap import Tap
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Union
from classes.config_loader import load_config
from classes.path_managment import StorePathHelper
from CustomLogging import setup_logging

# ----- logging -----

logger = setup_logging(logger=logging.getLogger(__name__))

# ----- tap -----

class Args(Tap):
    njets: Literal["all", "0", "1", "2"] = "all"

# ----- paths -----

config_path = "../configs/config_NF.yaml"
used_trainings_path = "../configs/paths1.yaml"

# ----- constants -----

PDF_EPS = 1e-300          
LOG_PDF_CLAMP = -700.0   
NUM_BINS_PDF = 500
NUM_BINS_FF = 500

QUANTILES = {
    "loose": 0.99,
    "tight": 0.98,
}
# ----- config -----

@dataclass
class ModelConfig:
    n_layers: int
    hidden_dims: int
    s_scale: float


# ----- classes -----


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


def load_model_config(path: str) -> ModelConfig:
    cfg = load_config(path)
    return ModelConfig(
        n_layers=cfg["model"]["n_layers"],
        hidden_dims=cfg["model"]["hidden_dims"],
        s_scale=cfg["model"]["s_scale"],
    )

# ----- data utilities -----

def create_4d_numpy_data(events) -> np.ndarray:
    """Canonical AR / SR representation."""
    return np.vstack(
        [events.pt_1, events.pt_2, events.m_vis, events.deltaR_ditaupair]
    ).T


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)

# ----- model utilities -----

def load_flow(
    dim: int,
    cfg: ModelConfig,
    checkpoint_path: str,
    device: torch.device,
) -> RealNVP_with_njets:

    model = RealNVP_with_njets(
        dim=dim,
        n_layers=cfg.n_layers,
        hidden_dims=(cfg.hidden_dims,),
        s_scale=cfg.s_scale,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

@torch.no_grad()
def evaluate_pdf(model: RealNVP_with_njets, X: torch.Tensor, nj_cls: torch.Tensor) -> np.ndarray:
    """Returns PDF evaluated at events"""
    log_pdf = model.log_prob(X, nj_cls)
    log_pdf = torch.clamp(log_pdf, min=LOG_PDF_CLAMP)
    return torch.exp(log_pdf).cpu().numpy()

# ----- FF logic -----

def compute_eventwise_fake_factors(
    pdf_AR: np.ndarray,
    pdf_SR: np.ndarray,
    global_ff: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:

    # ---- safe ratio
    ratio = np.divide(
        pdf_SR,
        np.maximum(pdf_AR, PDF_EPS),
        out=np.zeros_like(pdf_SR, dtype=float),
        where=(pdf_AR > 0) & (pdf_SR > 0)
)

    # ---- quantile clipping
    q = 0.99
    clip_value = np.quantile(ratio, q)
    global_ff_cor = global_ff / q

    # ---- full FF for all events
    ff_eventwise_full = global_ff_cor * ratio

    # ---- mask for clipping and nonzero PDFs
    clip_mask = (ratio <= clip_value) & (pdf_AR > 0) & (pdf_SR > 0)

    # ---- clipped FF
    ff_eventwise_clipped = ff_eventwise_full[clip_mask]

    return ff_eventwise_full, ff_eventwise_clipped, global_ff_cor, clip_mask


# ----- plotting -----

def plot_pdf_comparison(pdf_AR_like, pdf_SR_like, outdir: StorePathHelper):

    min_val = max(1e-12, min(pdf_AR_like.min(), pdf_SR_like.min()))
    max_val = max(pdf_AR_like.max(), pdf_SR_like.max())
    bins = np.logspace(np.log10(min_val), np.log10(max_val), NUM_BINS_PDF + 1)
    counts_AR_like, edges_AR_like = np.histogram(pdf_AR_like, bins=bins)
    counts_SR_like, edges_SR_like = np.histogram(pdf_SR_like, bins=bins)
    
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))

    ax[0].hist(pdf_SR_like, bins=bins, alpha=0.7, label="SR-like")
    ax[1].hist(pdf_AR_like, bins=bins, alpha=0.7, label="AR-like")

    for a in ax:
        a.set_xscale("log")
        a.legend()
        a.grid(True)

    fig.tight_layout()
    fig.savefig(str(outdir.autopath / "pdf.png"))
    fig.savefig(str(outdir.autopath / "pdf.pdf"))
    plt.close(fig)

def plot_ff_distribution(
    ff: np.ndarray,
    global_ff: float,
    global_ff_corr: float,
    outdir: StorePathHelper,
):
    bins = np.logspace(
        max(np.log10(ff.min()), np.log10(0.01)),
        np.log10(ff.max()),
        NUM_BINS_FF + 1,
    )

    mean_ff = np.mean(ff)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(ff, bins=bins, alpha = 0.7)
    ax.axvline(global_ff, label = f'global FF: {np.round(global_ff, 2)}')
    ax.axvline(global_ff_corr, label= f'corrected global FF: {np.round(global_ff_corr, 2)}')
    ax.axvline(mean_ff, label= f'mean FF {np.round(mean_ff, 2)}')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(str(outdir.autopath / 'ff.png'))
    fig.savefig(str(outdir.autopath / 'ff.pdf'))
    plt.close(fig)

def plot_ff_estimation(SR, SR_weights, AR, AR_weights, ff_eventwise_clipped, global_ff, clip_mask, plot_dir):

    AR_clipped = AR[clip_mask]
    AR_weights_clipped = AR_weights[clip_mask]

    print( len(ff_eventwise_clipped), np.shape(AR_clipped)[0])

    weights_fe = np.multiply(AR_weights_clipped, ff_eventwise_clipped)
    weights_global_ff = global_ff * AR_weights_clipped

    titles = [r'$p_T^\mu$', r'$p_T^{\tau_h}$', r'$m_{vis}$', r'$\Delta R$']
    titles_files = ['pt_1', 'pt_2', 'm_vis', 'deltaR']
    units = [' in GeV', ' in GeV', ' in GeV', '']
    xlims = [[25, 95], [30, 80], [0, 220], [0,4.2]]
    steps = [1,1,22/6,0.07]


    fig, ax = plt.subplots(
        2,4, 
        figsize = (18,5), 
        sharex='col',
        gridspec_kw={"height_ratios": [3,1], "hspace": 0.1})


    for i in range(4):

        bins = np.arange(xlims[i][0], xlims[i][1]+steps[i] ,  steps[i])

        counts_SR, bin_edges = np.histogram(SR.T[i], bins = bins,  weights = SR_weights)
        counts_flow, _ = np.histogram(AR_clipped.T[i], bins = bin_edges, weights = weights_fe)
        counts_globalFF, _ = np.histogram(AR_clipped.T[i], bins = bin_edges, weights = weights_global_ff)

        ratio_flow = counts_flow / np.where(counts_SR == 0, 1, counts_SR)
        ratio_globalFF = counts_globalFF / np.where(counts_SR == 0, 1, counts_SR)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # ---- plot only one variable ------

        fig2, ax2 = plt.subplots(
            2,1,
            figsize=(10,6),
            sharex = 'col',
            gridspec_kw={'height_ratios': [3,1], 'hspace': 0.1}
            )
        
        ax2[0].hist(SR.T[i], bins = bins, weights = SR_weights, color = 'orange', label = 'SR', alpha = 0.5)
        ax2[0].hist(AR_clipped.T[i], bins = bins, weights = weights_fe, color = 'blue', linewidth = 1.5, label = 'Flow FF estimation', histtype = 'step')
        ax2[0].hist(AR_clipped.T[i], bins = bins, weights = global_ff * AR_weights_clipped, color = 'red', linewidth = 1.5, label = 'Global FF estimation', histtype = 'step')
        ax2[0].set_xlim(xlims[i])
        ax2[0].set_ylabel('#')
        ax2[0].legend()
        ax2[0].grid(True)

        # ------ plot ratios ------

        ax2[1].grid(True)
        ax2[1].stairs(ratio_flow, bins, linestyle='-', color = 'blue', label = 'Flow FF estimation')
        ax2[1].stairs(ratio_globalFF, bins, linestyle='-', color = 'red', label = 'Global FF estimation')
        ax2[1].set_xlabel(titles[i]+units[i])
        ax2[1].set_ylabel('Ratio')
        ax2[1].axhline(1, color='gray', linestyle='--')
        ax2[1].set_ylim([0.5, 1.5])
        fig2.savefig(str(plot_dir.autopath.joinpath(titles_files[i])))
        fig2.savefig(str(plot_dir.autopath.joinpath(f'{titles_files[i]}.pdf')))

        # ---- plot all 4 plots in one figure --------

        # ----- plot hists ------

        ax[0, i].hist(SR.T[i], bins = bins, weights = SR_weights, color = 'orange', label = 'SR', alpha = 0.5)
        ax[0, i].hist(AR_clipped.T[i], bins = bins, weights = weights_fe, color = 'blue', label = 'Flow FF estimation', histtype = 'step')
        ax[0, i].hist(AR.T[i], bins = bins, weights = global_ff * AR_weights, color = 'red', label = 'Global FF estimation', histtype = 'step')
        ax[0, i].set_xlim(xlims[i])
        ax[0, i].set_ylabel('#')
        ax[0, i].grid(True)

        # ------ plot ratios ------

        ax[1, i].grid(True)
        ax[1, i].stairs(ratio_flow, bins, linestyle='-', color = 'blue', label = 'Flow FF estimation')
        ax[1, i].stairs(ratio_globalFF, bins, linestyle='-', color = 'red', label = 'Global FF estimation')
        ax[1, i].set_xlabel(titles[i]+units[i])
        ax[1, i].set_ylabel('Ratio')
        ax[1, i].axhline(1, color='gray', linestyle='--')
        ax[1, i].set_ylim([0.5, 1.5])

    handles, labels = ax[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper right",
        ncol=2,
        frameon=False
    )
    fig.tight_layout(rect=[0,0,1,0.9])
    fig.subplots_adjust(left=0.06, right = 0.98)
    fig.subplots_adjust(wspace = 0.25)
    fig.savefig(str(plot_dir.autopath.joinpath('ff_estimation.png')))
    fig.savefig(str(plot_dir.autopath.joinpath('ff_estimation.pdf')))

# ----- main -----

def main():

    args = Args().parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f'Using device: {device}')

    # --- configs ---

    model_cfg = load_model_config(config_path)
    paths_cfg = load_config(used_trainings_path)

    path_AR = paths_cfg["training_results"]['all']["AR-like"]
    path_SR = paths_cfg["training_results"]['all']["SR-like"]

    # --- data ---

    data_AR_like = pd.read_feather('../data/data_AR_like.feather')
    data_SR_like = pd.read_feather('../data/data_SR_like.feather')
    data_AR = pd.read_feather('../data/data_AR.feather')
    data_SR = pd.read_feather('../data/data_SR.feather')

    data_AR_like_pt = torch.load('../data/data_AR_like.pt')

    SR = create_4d_numpy_data(data_SR)
    SR_njets = data_SR.njets
    SR_weights = data_SR.scl_w
    AR = create_4d_numpy_data(data_AR)
    AR_njets = data_AR.njets
    AR_weights = data_AR.scl_w

    njets = AR_njets.to_numpy()


    # --- masking for njets ----
    
    if args.njets == "all":
        AR = AR
        AR_weights = AR_weights
        SR = SR
        SR_weights = SR_weights
    elif args.njets == '0':
        AR = AR[AR_njets == 0]
        AR_weights = AR_weights[AR_njets == 0]
        SR = SR[SR_njets == 0]
        SR_weights = SR_weights[SR_njets == 0]
        njets = njets[AR_njets == 0]
    elif args.njets == '1':
        AR = AR[AR_njets == 1]
        AR_weights = AR_weights[AR_njets == 1]
        SR = SR[SR_njets == 1]
        SR_weights = SR_weights[SR_njets == 1]
        njets = njets[AR_njets == 1]
    elif args.njets == '2':
        AR = AR[AR_njets >= 2]
        AR_weights = AR_weights[AR_njets >= 2]
        SR = SR[SR_njets >= 2]
        SR_weights = SR_weights[SR_njets >= 2]
        njets = njets[AR_njets >= 2]
    else:
        raise ValueError(f"{args.njets} no valid input")

    events = to_tensor(AR, device)
    njets = to_tensor(njets, device)
    n_jcls = map_njets_to_3cls(njets)
    dim = AR.shape[1]

    logger.info("Loaded %d AR events (dim=%d)", len(AR), dim)

    # --- models ---

    model_AR = load_flow(dim, model_cfg, path_AR, device)
    model_SR = load_flow(dim, model_cfg, path_SR, device)

    # --- pfd's ---

    

    pdf_AR = evaluate_pdf(model_AR, events, n_jcls)
    pdf_SR = evaluate_pdf(model_SR, events, n_jcls)

    # --- ff ---

    global_ff = len(data_SR_like[(data_SR_like.pt_2 > 30) &  (data_SR_like.pt_1 > 25)]) / len(data_AR_like[(data_AR_like.pt_2 > 30) &  (data_AR_like.pt_1 > 25)])

    ff_eventwise_full, ff_eventwise_clipped, global_ff_corr, clip_mask  = compute_eventwise_fake_factors(
        pdf_AR,
        pdf_SR,
        global_ff,
    )

    AR_clipped = AR[clip_mask]
    AR_weights_clipped = AR_weights[clip_mask]


    assert len(ff_eventwise_clipped) == len(AR_clipped)

    # --- save FF ---

    ff_dir = StorePathHelper(directory=f"../data/FF/{args.njets}")
    np.save(str(ff_dir.autopath / 'FF_event_wise.npy'), ff_eventwise_full)

    # --- plots ---
    
    plot_dir = StorePathHelper(directory=f'FF_results/{args.njets}')
    plot_pdf_comparison(pdf_AR, pdf_SR, plot_dir)
    plot_ff_distribution(ff_eventwise_clipped, global_ff, global_ff_corr, plot_dir)
    plot_ff_estimation(SR, SR_weights, AR, AR_weights, ff_eventwise_clipped, global_ff_corr, clip_mask, plot_dir)

# -----------

if __name__ == "__main__":
    main()