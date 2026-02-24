from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tap import Tap

from classes.config_loader import load_config
from classes.models import RealNVP
from classes.dataloading import Datasets
from classes.path_managment import StorePathHelper
from CustomLogging import setup_logging

# ----- logging -----

logger = setup_logging(logger=logging.getLogger(__name__))

# ----- tap -----

class Args(Tap):
    preselection: Literal["loose", "tight"] = "loose"
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
        [events.pt_1, events.pt_2, events.m_vis, events.deltaR]
    ).T


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)

# ----- model utilities -----

def load_flow(
    dim: int,
    cfg: ModelConfig,
    checkpoint_path: str,
    device: torch.device,
) -> RealNVP:

    model = RealNVP(
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
def evaluate_pdf(model: RealNVP, events: torch.Tensor) -> np.ndarray:
    """Returns PDF evaluated at events"""
    log_pdf = model.log_prob(events)
    log_pdf = torch.clamp(log_pdf, min=LOG_PDF_CLAMP)
    return torch.exp(log_pdf).cpu().numpy()

# ----- FF logic -----

def compute_eventwise_fake_factors(
    pdf_AR: np.ndarray,
    pdf_SR: np.ndarray,
    global_ff: float,
    preselection: Literal["loose", "tight"],
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:

    # ---- safe ratio
    ratio = np.divide(
        pdf_SR,
        np.maximum(pdf_AR, PDF_EPS),
        out=np.zeros_like(pdf_SR, dtype=float),
        where=(pdf_AR > 0) & (pdf_SR > 0)
)

    # ---- quantile clipping
    q = QUANTILES[preselection]
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
    steps = [5,5,110/6,0.35]


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

    preselection = args.preselection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f'Using device: {device}')

    # --- configs ---

    model_cfg = load_model_config(config_path)
    paths_cfg = load_config(used_trainings_path)

    path_AR = paths_cfg["training_results"][str(args.njets)][preselection]["AR-like"]
    path_SR = paths_cfg["training_results"][str(args.njets)][preselection]["SR-like"]

    # --- data ---

    data = Datasets()
    SR = create_4d_numpy_data(data.SR_tight)
    SR_njets = data.SR_tight.njets
    SR_weights = data.SR_tight.weight
    AR = create_4d_numpy_data(data.AR_tight)
    AR_njets = data.AR_tight.njets
    AR_weights = data.AR_tight.weight


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
    elif args.njets == '1':
        AR = AR[AR_njets == 1]
        AR_weights = AR_weights[AR_njets == 1]
        SR = SR[SR_njets == 1]
        SR_weights = SR_weights[SR_njets == 1]
    elif args.njets == '2':
        AR = AR[AR_njets >= 2]
        AR_weights = AR_weights[AR_njets >= 2]
        SR = SR[SR_njets >= 2]
        SR_weights = SR_weights[SR_njets >= 2]
    else:
        raise ValueError(f"{args.njets} no valid input")

    events = to_tensor(AR, device)
    dim = AR.shape[1]

    logger.info("Loaded %d AR events (dim=%d)", len(AR), dim)

    # --- models ---

    model_AR = load_flow(dim, model_cfg, path_AR, device)
    model_SR = load_flow(dim, model_cfg, path_SR, device)

    # --- pfd's ---

    pdf_AR = evaluate_pdf(model_AR, events)
    pdf_SR = evaluate_pdf(model_SR, events)

    # --- ff ---

    global_ff = len(data.SR_like_tight) / len(data.AR_like_tight)

    ff_eventwise_full, ff_eventwise_clipped, global_ff_corr, clip_mask  = compute_eventwise_fake_factors(
        pdf_AR,
        pdf_SR,
        global_ff,
        preselection
    )

    AR_clipped = AR[clip_mask]
    AR_weights_clipped = AR_weights[clip_mask]


    assert len(ff_eventwise_clipped) == len(AR_clipped)

    # --- save FF ---

    ff_dir = StorePathHelper(directory=f"../data/FF/{args.njets}/{preselection}")
    np.save(str(ff_dir.autopath / 'FF_event_wise.npy'), ff_eventwise_full)

    # --- plots ---
    
    plot_dir = StorePathHelper(directory=f'FF_results/{args.njets}/{preselection}')
    plot_pdf_comparison(pdf_AR, pdf_SR, plot_dir)
    plot_ff_distribution(ff_eventwise_clipped, global_ff, global_ff_corr, plot_dir)
    plot_ff_estimation(SR, SR_weights, AR, AR_weights, ff_eventwise_clipped, global_ff_corr, clip_mask, plot_dir)

# -----------

if __name__ == "__main__":
    main()