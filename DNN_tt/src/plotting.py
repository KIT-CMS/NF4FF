from pathlib import Path
import logging
import random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tap import Tap
import torch as t
from typing import Literal
import uproot

from classes.Plotting import plot_fake_factors, plot_closure, plot_fake_factors_grouped, plot_fake_factors_in_dr_grouped
from classes.Loading import load_config, load_labels, load_data
from classes.Plotting import FF_closure_in_DR_tau1, plot_fake_factors_grouped_combTaus, plot_classic_fake_factors, plot_fake_factors_incl, plot_closure_incl



SEED = 42
logger = logging.getLogger(__name__)


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

class Args(Tap):
    embedding: Literal["embedding", "no_embedding"] = "embedding"
    var = "variables"

    taus: Literal['split', 'incl'] = 'incl' # split: calc 2 FF for tau1 and tau2 | incl: calc only 1 FF
    dnn_grouped: bool = False
    classic: bool = False

    closure_DR: bool = False
    FF_dist: bool = True
    closure_AR: bool = True

args = Args().parse_args()

cfg_path = load_config('/work/tapp/TauFF/NF4FF/DNN_tt/configs/config_path.yaml')

DATA_PATH = f'{cfg_path["datasets"]}/{args.embedding}/combined_data_updated.feather'
DATA_CLASSIC_JV_PATH = "/work/tapp/TauFF/NF4FF/Data/datasets/classic/combined_data_jvoss.feather"
DATA_CLASSIC_SG_PATH = "/work/tapp/TauFF/NF4FF/Data/datasets/classic/combined_data_sgiappic.feather"
MASKS_PATH = cfg_path["masks"]
MASKS_PATH_INCL = cfg_path["masks_incl"]
TRAINING_VAR_PATH = cfg_path["train_var"]
NN_CONFIG_PATH = cfg_path["DNN"]
CHECKPOINT_DIR = cfg_path["traininfg_results"]

PLOTTING_CONFIG_PATH = cfg_path["cfg_plotting"]
LABELS_CONFIG_PATH = cfg_path["labels"]

#PLOTS_DIR = Path('../plots/layers_3/ReLU')
PLOTS_DIR = Path(cfg_path["plots"])
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_GROUPINGS = ('njets', 'tau_decaymode', 'ungrouped', 'classic')
PLOT_SUBDIRS = ('closure_in_DR', 'FF_distribution_AR', 'FF_distribution_DR', 'closure_plots')
for subdir in PLOT_SUBDIRS:
    for grouping in PLOT_GROUPINGS:
        (PLOTS_DIR / subdir / grouping).mkdir(parents=True, exist_ok=True)

PLOT_GROUPINGS = ('njets', 'tau_decaymode')


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
    df_incl = load_data(DATA_PATH, MASKS_PATH_INCL)
    df_classic_jv = load_data(DATA_CLASSIC_JV_PATH, MASKS_PATH)
    df_classic_sg = load_data(DATA_CLASSIC_SG_PATH, MASKS_PATH)
    #print(df_incl.columns)
    #exit()


    # ----- Closure plots in DR -----
    if args.closure_DR:
        if args.dnn_grouped and args.taus=='split':
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

        elif args.dnn_grouped and args.taus=='incl':
            logger.warning('Tau inclusive grouped DNN is not yet implemented.')
        
        elif not args.dnn_grouped and args.taus=='split':
            logger.warning('Not yet implemented.')

        elif not args.dnn_grouped and args.taus=='incl':
            for var in VARIABLES_SMALL:
                bins, label = get_bins_and_label(var)

                fig_q, _ = FF_closure_in_DR_tau1(
                    df=df,
                    var=var,
                    bins=bins,
                    label=label,
                    grouping=grouping,
                )
                plt.savefig(PLOTS_DIR / 'closure_in_DR' / 'ungrouped' / f'FF_closure_DR_tau1_{var}.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'closure_in_DR' / 'ungrouped' / f'FF_closure_DR_tau1_{var}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig_q)

    # ----- Fake-factor distributions -----
    if args.FF_dist:
        if args.dnn_grouped and args.taus=='split':
            # ----- clipped FF -----
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


                # ----- unclipped FF -----
                fig_ar, ax_ar = plot_fake_factors_grouped(
                    df=df,
                    category_title=f'split in {grouping}',
                    grouping=grouping,
                    clipped=False
                )
                plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / grouping / f'plot_ff_unclipped_splitTaus_{grouping}.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / grouping / f'plot_ff_unclipped_splitTaus_{grouping}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig_ar)
                logger.info(f'Saved FF distributions in AR for {grouping}')

        elif args.dnn_grouped and args.taus=='incl':
            logger.warning('Tau inclusive grouped DNN is not yet implemented.')

        elif not args.dnn_grouped and args.taus=='split':
            # ----- DNN FF -----
            fig_ar, ax_ar = plot_fake_factors(df=df)
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / 'ungrouped' / f'plot_ff_splitTaus.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / 'ungrouped' / f'plot_ff_splitTaus.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved FF distributions in AR for ungrouped DNN')

            fig_ar, ax_ar = plot_fake_factors(df=df, clipped=False)
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / 'ungrouped' / f'plot_ff_unclipped_splitTaus.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / 'ungrouped' / f'plot_ff_unclipped_splitTaus.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved unclipped FF distributions in AR for ungrouped DNN')
        
        elif not args.dnn_grouped and args.taus=='incl':
            # ----- DNN FF -----
            fig_ar, ax_ar = plot_fake_factors_incl(df=df_incl)
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / 'ungrouped' / f'plot_ff_incl.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / 'ungrouped' / f'plot_ff_incl.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved inclusive FF distributions in AR for ungrouped DNN')

            fig_ar, ax_ar = plot_fake_factors_incl(df=df_incl, clipped=False)
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / 'ungrouped' / f'plot_ff_unclipped_incl.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / 'ungrouped' / f'plot_ff_unclipped_incl.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved inclusive unclipped FF distributions in AR for ungrouped DNN')

        if args.classic:
            # ----- classic FF -----
            fig_ar, ax_ar = plot_classic_fake_factors(df=df_classic_jv, short='jv')
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / 'classic' / f'plot_jv_ff_splitTaus.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / 'classic' / f'plot_jv_ff_splitTaus.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)

            fig_ar, ax_ar = plot_classic_fake_factors(df=df_classic_sg, short='sg')
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / 'classic' / f'plot_sg_ff_splitTaus.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / 'classic' / f'plot_sg_ff_splitTaus.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved FF distributions in AR for classic')



    if args.closure_AR:

        x = uproot.open("/work/ptoedter/MA-Pascal/smhtt_ul/output/2018-et-2025-12_15_with_uncertainties_ntupels_v1-final_v3_2026_02_19/control_shapes-2018-et-2025-12_15_with_uncertainties_ntupels_v1-final_v3_2026_02_19.root")
        bkgs = [it for it in x.keys() if "#q_1;" in it and "Nominal" in it and any(subit in it for subit in ["TT-TTL", "DY-ZL", "jetFakes#", "VV-VVL", "EMB#"])]
        corr_emb_ff = sum([x[it].to_numpy()[0] for it in bkgs]) / [x[next(it for it in x.keys() if "data" in it and "Nominal" in it and "#q_1" in it)].to_numpy()[0]]

        if args.dnn_grouped and args.taus=='split':
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

            logger.info('Saved all closure plots in tau decaymode')


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

        elif args.dnn_grouped and args.taus=='incl':
            logger.warning('Tau inclusive grouped DNN is not yet implemented.')

        elif not args.dnn_grouped and args.taus=='split':
            for var in VARIABLES_SMALL:
                bins, label = get_bins_and_label(var)
                fig, ax, _ = plot_closure(
                        df = df,
                        var = var,
                        bins = bins,
                        label = label,
                        grouping = None,
                        corr_emb_ff = corr_emb_ff,
                    )
                
                plt.savefig(PLOTS_DIR / 'closure_plots' / 'ungrouped' / f'plot_closure_{var}.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'closure_plots' / 'ungrouped' / f'plot_closure_{var}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)

            logger.info('Saved all closure plots for ungrouped DNN')

        elif not args.dnn_grouped and args.taus=='incl':
            for var in VARIABLES_SMALL:
                bins, label = get_bins_and_label(var)
                fig, ax, _ = plot_closure_incl(
                        df = df_incl,
                        var = var,
                        bins = bins,
                        label = label,
                        grouping = None,
                        corr_emb_ff = corr_emb_ff,
                    )
                
                incl_plot_dir = PLOTS_DIR / 'closure_plots' / 'ungrouped' / 'tau_incl'
                incl_plot_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(incl_plot_dir / f'plot_closure_{var}.png', dpi=150, bbox_inches='tight')
                plt.savefig(incl_plot_dir / f'plot_closure_{var}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)

            logger.info('Saved all closure plots for tau inclusive ungrouped DNN')


# -------------------------------------------

if __name__ == '__main__':
    main()