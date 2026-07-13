from pathlib import Path
import logging
import random

import numpy as np
import matplotlib.pyplot as plt
import torch as t
import pandas as pd
import correctionlib as cr
from pathlib import Path
import matplotlib
import yaml
import uproot
from typing import Literal
from tap import Tap

from classes import load_variables, load_data, load_model, load_fold_combined_model, test_data
from classes import calculate_fake_factors, calculate_fake_factor_dnn, calculate_fake_factor_classic, calculate_fake_factors_in_DR_wjets, calculate_fake_factors_in_DR_qcd
from classes import plot_fake_factors_in_DR, plot_fake_factors
from classes import CMS_CHANNEL_TITLE, CMS_CATEGORY_TITLE, CMS_LUMI_TITLE, CMS_LABEL, adjust_ylim_for_legend, plot_closure, plot_fake_factors_grouped, plot_fake_factors_in_dr_grouped
from classes.Loading import load_config, load_variables, load_labels
from classes.Plotting import FF_closure_in_DR_tau1, plot_fake_factors_grouped_combTaus



SEED = 42
logger = logging.getLogger(__name__)


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

class Args(Tap):
    taus = [1, 2] #[1, 2, 12] # list of tau fakes
    embedding: Literal["embedding", "no_embedding"] = "embedding"
    var = "variables_61"
    closure_DR: bool = False
    FF_dist: bool = True
    closure_AR: bool = False

args = Args().parse_args()

cfg_path = load_config('/work/tapp/TauFF/NF4FF/DNN_tt/configs/config_path.yaml')

DATA_PATH = f'{cfg_path["datasets"]}/{args.embedding}/combined_data_updated.feather'
MASKS_PATH = cfg_path["masks"]
TRAINING_VAR_PATH = cfg_path["train_var"]
NN_CONFIG_PATH = cfg_path["DNN"]
CHECKPOINT_DIR = cfg_path["traininfg_results"]

PLOTTING_CONFIG_PATH = cfg_path["cfg_plotting"]
LABELS_CONFIG_PATH = cfg_path["labels"]

#PLOTS_DIR = Path('../plots/layers_3/ReLU')
PLOTS_DIR = Path(cfg_path["plots"])
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_GROUPINGS = ('njets', 'tau_decaymode')
PLOT_SUBDIRS = ('closure_in_DR', 'FF_distribution_AR', 'FF_distribution_DR', 'closure_plots')
for subdir in PLOT_SUBDIRS:
    for grouping in PLOT_GROUPINGS:
        (PLOTS_DIR / subdir / grouping).mkdir(parents=True, exist_ok=True)


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
    'ytick.right': True,
})

PLOTTING_CFG = load_config(PLOTTING_CONFIG_PATH)
LABELS_CFG = load_labels(LABELS_CONFIG_PATH)

VARIABLES_SMALL = PLOTTING_CFG.get('variables_set_small', [])
VARIABLES_LARGE = PLOTTING_CFG.get('variables_set_large', [])


def get_bins(variable):
    bin_spec = PLOTTING_CFG.get('bins_by_variable', {}).get(variable)
    if bin_spec is None:
        raise KeyError(f'No bin specification found for variable: {variable}')

    if isinstance(bin_spec, (list, tuple)) and len(bin_spec) == 3:
        start, stop, num = bin_spec
        return np.linspace(float(start), float(stop), int(num))

    return np.asarray(bin_spec, dtype=float)


def get_label(variable, channel='et'):
    labels_by_channel = LABELS_CFG.get(channel, {}) if isinstance(LABELS_CFG, dict) else {}
    return labels_by_channel.get(variable, variable)


def get_bins_and_label(variable, channel='et'):
    return get_bins(variable), get_label(variable, channel)


# ------------------------------------------------------

def main():
    df = load_data(DATA_PATH, MASKS_PATH)
    print(df.columns)


    # ----- Closure plots in DR -----
    if args.closure_DR:
        for grouping in PLOT_GROUPINGS:
            for var in VARIABLES_SMALL:
                bins, label = get_bins_and_label(var)

                fig_q, _ = FF_closure_in_DR_tau1(
                    df=df,
                    var=var,
                    bins=bins,
                    label=label,
                    grouping=grouping,
                )
                plt.savefig(PLOTS_DIR / 'closure_in_DR' / grouping / f'FF_closure_DR_tau1_{var}.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'closure_in_DR' / grouping / f'FF_closure_DR_tau1_{var}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig_q)

            logger.info(f'Saved closure plots in DR for {grouping}')


    # ----- Fake-factor distributions -----
    if args.FF_dist:
        for grouping in PLOT_GROUPINGS:
            fig_ar, ax_ar = plot_fake_factors_grouped(
                df=df,
                category_title=f'split in {grouping}',
                grouping=grouping,
            )
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / grouping / f'plot_ff_splitTaus_{grouping}.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / grouping / f'plot_ff_splitTaus_{grouping}.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved FF distributions in AR for {grouping}')


            fig_ar_ct, ax_ar_ct = plot_fake_factors_grouped_combTaus(
                df=df,
                category_title=f'split in {grouping}',
                grouping=grouping,
            )
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / grouping / f'plot_ff_combTaus_{grouping}.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / grouping / f'plot_ff_combTaus_{grouping}.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar_ct)
            logger.info(f'Saved FF distributions in AR for combined Taus for {grouping}')

            fig_dr, ax_dr = plot_fake_factors_in_dr_grouped(
                df=df,
                category_title=f'split in {grouping}',
                grouping=grouping,
            )
            plt.savefig(PLOTS_DIR / 'FF_distribution_DR' / grouping / f'plot_ff_DR_splitTaus_{grouping}.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'FF_distribution_DR' / grouping / f'plot_ff_DR_splitTaus_{grouping}.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_dr)
            logger.info(f'Saved FF distributions in DR for {grouping}')
            exit()

    if args.closure_AR:

        x = uproot.open("/work/ptoedter/MA-Pascal/smhtt_ul/output/2018-et-2025-12_15_with_uncertainties_ntupels_v1-final_v3_2026_02_19/control_shapes-2018-et-2025-12_15_with_uncertainties_ntupels_v1-final_v3_2026_02_19.root")
        bkgs = [it for it in x.keys() if "#q_1;" in it and "Nominal" in it and any(subit in it for subit in ["TT-TTL", "DY-ZL", "jetFakes#", "VV-VVL", "EMB#"])]
        corr_emb_ff = sum([x[it].to_numpy()[0] for it in bkgs]) / [x[next(it for it in x.keys() if "data" in it and "Nominal" in it and "#q_1" in it)].to_numpy()[0]]


        for var in VARIABLES_SMALL:
            bins, label = get_bins_and_label(var)
            fig, ax, _ = plot_closure(
                    df = df,
                    var = var,
                    bins = bins,
                    label = label,
                    grouping = 'tau_decaymode',
                    corr_emb_ff = corr_emb_ff,
                )
            
            plt.savefig(PLOTS_DIR / 'closure_plots' / 'tau_decaymode' / f'plot_closure_{var}.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'closure_plots' / 'tau_decaymode' / f'plot_closure_{var}.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig)

        print('Saved all closure plots in tau decaymode')


        for var in VARIABLES_SMALL:
            bins, label = get_bins_and_label(var)
            fig, ax, _ = plot_closure(
                    df = df,
                    var = var,
                    bins = bins,
                    label = label,
                    grouping = 'njets',
                    corr_emb_ff = corr_emb_ff,
                )
            
            plt.savefig(PLOTS_DIR / 'closure_plots' / 'njets' / f'plot_closure_{var}.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'closure_plots' / 'njets' / f'plot_closure_{var}.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig)

        logger.info('Saved all closure plots in njets')

# -------------------------------------------

if __name__ == '__main__':
    main()