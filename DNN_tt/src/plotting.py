from pathlib import Path
import logging
import random
import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tap import Tap
import torch as t
from typing import Literal
import uproot

from classes.Loading import load_config, load_labels, load_data, load_variables
from classes.Fraction_factor import fraction_in_bins, fraction_in_bins_grouped, fraction_in_bins_3split, fraction_in_bins_grouped_3split
from classes.Plotting import plot_fake_factors, plot_closure, plot_fake_factors_grouped, plot_fake_factors_in_dr_grouped
from classes.Plotting import FF_closure_in_DR_tau1, FF_closure_in_DR_tau2, FF_closure_in_DR_incl, plot_fake_factors_grouped_combTaus, plot_classic_fake_factors, plot_fake_factors_incl, plot_closure_incl, plot_fake_factors_combTaus, plot_fake_factors_grouped_incl
from classes.Plotting import plot_closure_3split, FF_closure_in_DR_3split, plot_fake_factors_3split, plot_fake_factors_combTaus_3split, plot_fake_factors_grouped_3split, plot_fake_factors_grouped_combTaus_3split
from classes.Plotting import plot_fractions, plot_fractions_grouped, plot_fractions_3split, plot_fractions_grouped_3split
from classes.Plotting import plot_in_hist1d, plot_NN_output_frac, plot_NN_output_frac_stacked, plot_NN_output_frac_overlay, plot_NN_output_frac_ratio, plot_NN_output_frac_events
from classes.Plotting import plot_closure_MC, validate_nf_weights

SEED = 42
logger = logging.getLogger(__name__)


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

class Args(Tap):
    embedding: Literal["embedding", "no_embedding"] = "embedding"
    var = "variables"

    taus: Literal['split', 'incl', '3split'] = 'split' # split: calc 2 FF for tau1 and tau2 | incl: calc only 1 FF
    incl: Literal['and', 'or', 'andor'] = 'andor' # Combine tau1 and tau2 AR with and or or
    fraction_max_bins: int = 0 # Maximum displayed bins per axis for 3split fractions; 0 keeps all bins
    dnn_grouped: bool = True

    AR_comb: bool = False
    closure_MC: bool = False
    classic: bool = False

    mode: Literal['dnn', 'nf'] = 'nf'
    wo_train: bool = False
    vars: Literal['small', 'large'] = 'small' # small: use small set of variables | large: use large set of variables

    closure_DR: bool = False
    FF_dist: bool = True
    frac: Literal['global', 'pt_binned', 'DNN'] = 'global' # global: use global fraction | pt_binned: use pt-binned fraction | DNN: use DNN-based fraction
    closure_AR: bool = True
    

args = Args().parse_args()

cfg_path = load_config('/work/tapp/TauFF/NF4FF/DNN_tt/configs/config_path.yaml')

DATA_PATH = f'{cfg_path["datasets"]}/{args.embedding}/combined_data_updated.feather'
DATA_PATH_NOEMBED = f'{cfg_path["datasets"]}/no_embedding/combined_data.feather'
DATA_CLASSIC_JV_PATH = "/work/tapp/TauFF/NF4FF/Data/datasets/classic/combined_data_jvoss.feather"
DATA_CLASSIC_SG_PATH = "/work/tapp/TauFF/NF4FF/Data/datasets/classic/combined_data_sgiappic.feather"

MASKS_PATH = cfg_path["masks"]
MASKS_MC = cfg_path["masks_mc"]
MASKS_PATH_INCL = [cfg_path["masks_incl_and"], cfg_path["masks_incl_or"], cfg_path["masks_incl_andor"]]
MASKS_PATH_3SPLIT = cfg_path["masks_3split"]

PLOTTING_CONFIG_PATH = cfg_path["cfg_plotting"]
LABELS_CONFIG_PATH = cfg_path["labels"]

TRAINING_VAR_PATH = cfg_path["train_var"]

if args.mode=='dnn': PLOTS_DIR = Path(cfg_path["plots"])
if args.mode=='nf' and args.taus=='split': PLOTS_DIR = Path('/work/tapp/TauFF/NF4FF/Normalizing_Flow_tt/plots')
else: 'Not implemented'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_GROUPINGS = ('njets', 'tau_decaymode', 'ungrouped', 'classic')
PLOT_SUBDIRS = ('closure_in_DR', 'FF_distribution_AR', 'FF_distribution_DR', 'control_plots')
PLOT_FRAC_SUBDIRS = ('global_fraction', 'pt_binned_fraction', 'DNN_fraction')
PLOT_VAR = ('pt1_pt2', 'mvis_pt2')
PLOT_REGIONS = ('tau1', 'tau2', 'tau1&2')
for subdir in PLOT_SUBDIRS:
    for grouping in PLOT_GROUPINGS:
        (PLOTS_DIR / f'tau_incl_{args.incl}' / subdir / grouping).mkdir(parents=True, exist_ok=True)
        for frac_dir in PLOT_FRAC_SUBDIRS:
            for plotvar in PLOT_VAR:
                for regdir in PLOT_REGIONS:
                    (PLOTS_DIR / 'tau_split' / subdir / grouping / frac_dir).mkdir(parents=True, exist_ok=True)
                    (PLOTS_DIR / 'tau_split' / 'Fraction_factors' / grouping / frac_dir / plotvar/ regdir).mkdir(parents=True, exist_ok=True)                
                    (PLOTS_DIR / 'tau_3split' / subdir / grouping / frac_dir).mkdir(parents=True, exist_ok=True)
                    (PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / grouping / frac_dir / plotvar / regdir).mkdir(parents=True, exist_ok=True)
                shutil.rmtree(PLOTS_DIR / 'tau_split' / 'Fraction_factors' / grouping / frac_dir / plotvar / 'tau1&2')
        shutil.rmtree(PLOTS_DIR / 'tau_split' / 'Fraction_factors' / grouping / 'global_fraction')
        shutil.rmtree(PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / grouping / 'global_fraction')



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


def get_label(variable, channel='tt'):
    labels_by_channel = LABELS_CFG.get(channel, {}) if isinstance(LABELS_CFG, dict) else {}
    return labels_by_channel.get(variable, variable)


def get_bins_and_label(variable, channel='tt'):
    return get_bins(variable), get_label(variable, channel)


# ------------------------------------------------------

def main():
    labels_cfg = load_config(LABELS_CONFIG_PATH)
    training_variables_multiclass = load_variables(TRAINING_VAR_PATH, 'variables_for_multiclass')

    if args.vars == 'small': variables = VARIABLES_SMALL
    elif args.vars == 'large': variables = VARIABLES_LARGE
    else: variables = VARIABLES_SMALL

    # ----- grouping definitions -----    
    grouping_njets = (
        (0,),
        (1,),
        (2, 1000),
    )

    grouping_tdm = (
        (0,),
        (1,),
        (10,),
        (11,),
    )    

    if args.classic:
        logger.info('Initialize plotting for classic FF...')

        df_classic_jv = load_data(DATA_CLASSIC_JV_PATH, MASKS_PATH)
        df_classic_sg = load_data(DATA_CLASSIC_SG_PATH, MASKS_PATH)

        # ----- classic FF -----
        fig_ar, ax_ar = plot_classic_fake_factors(df=df_classic_jv, short='jv', corr=False)
        plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / 'classic' / f'plot_jv_ff_splitTaus.png', dpi=150, bbox_inches='tight')
        plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / 'classic' / f'plot_jv_ff_splitTaus.pdf', dpi=150, bbox_inches='tight')
        plt.close(fig_ar)

        fig_ar, ax_ar = plot_classic_fake_factors(df=df_classic_jv, short='jv', corr=True)
        plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / 'classic' / f'plot_jv_corr_ff_splitTaus.png', dpi=150, bbox_inches='tight')
        plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / 'classic' / f'plot_jv_corr_ff_splitTaus.pdf', dpi=150, bbox_inches='tight')
        plt.close(fig_ar)

        fig_ar, ax_ar = plot_classic_fake_factors(df=df_classic_sg, short='sg', corr=False)
        plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / 'classic' / f'plot_sg_ff_splitTaus.png', dpi=150, bbox_inches='tight')
        plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / 'classic' / f'plot_sg_ff_splitTaus.pdf', dpi=150, bbox_inches='tight')
        plt.close(fig_ar)

        fig_ar, ax_ar = plot_classic_fake_factors(df=df_classic_sg, short='sg', corr=True)
        plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / 'classic' / f'plot_sg_corr_ff_splitTaus.png', dpi=150, bbox_inches='tight')
        plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / 'classic' / f'plot_sg_corr_ff_splitTaus.pdf', dpi=150, bbox_inches='tight')
        plt.close(fig_ar)
        logger.info(f'Saved FF distributions in AR for classic')

        return None

    if args.closure_MC:
        logger.info('Plotting control plots without fake factors')

        df = load_data(DATA_PATH, MASKS_PATH)

        for var in variables:
            bins, label = get_bins_and_label(var)
            label = labels_cfg['tt'][var]
            fig, ax, _ = plot_closure_MC(
                    df = df,
                    var = var,
                    bins = bins,
                    label = label,
                    embedding=True,
                    yield_corr=1.37
                )

            mc_dir = PLOTS_DIR / 'without_FF' / 'control_plots' / 'MC_embedding'
            mc_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(mc_dir / f'plot_closure_{var}.png', dpi=150, bbox_inches='tight')
            plt.savefig(mc_dir / f'plot_closure_{var}.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig)


        logger.info('Plotting control plots without fake factors')
        df = load_data(DATA_PATH_NOEMBED, MASKS_MC)
        
        for var in variables:
            bins, label = get_bins_and_label(var)
            label = labels_cfg['tt'][var]
            fig, ax, _ = plot_closure_MC(
                    df = df,
                    var = var,
                    bins = bins,
                    label = label,
                    embedding=False,
                    yield_corr=1.37
                )

            mc_dir = PLOTS_DIR / 'without_FF' / 'control_plots' / 'MC_only'
            mc_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(mc_dir / f'plot_closure_{var}.png', dpi=150, bbox_inches='tight')
            plt.savefig(mc_dir / f'plot_closure_{var}.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig)
        return None

    

    # ----- tau split FF -----
        
    if args.taus=='split' and args.dnn_grouped:
        logger.info('Initiaize plotting for tau split FF calculated through grouped DNN...')

        df = load_data(DATA_PATH, MASKS_PATH)
        if args.mode == 'nf':
            validate_nf_weights(df, grouping='njets')
        #print(list(df.columns))
        #exit()

        # ----- Closure plots in DR -----
        if args.closure_DR and args.mode=='dnn':
            for grouping in PLOT_GROUPINGS:
                for var in variables:
                    bins, label = get_bins_and_label(var)
                    label = labels_cfg['tt'][var]

                    fig_q, _ = FF_closure_in_DR_tau1(
                        df=df,
                        var=var,
                        bins=bins,
                        label=label,
                        grouping=grouping,
                    )
                    plt.savefig(PLOTS_DIR / 'tau_split' / 'closure_in_DR' / grouping / f'{args.frac}_fraction'/ f'FF_closure_DR_tau1_{var}.png', dpi=150, bbox_inches='tight')
                    plt.savefig(PLOTS_DIR / 'tau_split' / 'closure_in_DR' / grouping / f'{args.frac}_fraction'/ f'FF_closure_DR_tau1_{var}.pdf', dpi=150, bbox_inches='tight')
                    plt.close(fig_q)

                    fig_q, _ = FF_closure_in_DR_tau2(
                        df=df,
                        var=var,
                        bins=bins,
                        label=label,
                        grouping=grouping,
                    )
                    plt.savefig(PLOTS_DIR / 'tau_split' / 'closure_in_DR' / grouping / f'{args.frac}_fraction'/ f'FF_closure_DR_tau2_{var}.png', dpi=150, bbox_inches='tight')
                    plt.savefig(PLOTS_DIR / 'tau_split' / 'closure_in_DR' / grouping / f'{args.frac}_fraction'/ f'FF_closure_DR_tau2_{var}.pdf', dpi=150, bbox_inches='tight')
                    plt.close(fig_q)

                logger.info(f'Saved closure plots in DR for {grouping}')

        # ----- Fake-factor distributions -----
        if args.FF_dist:
            
            for grouping, group_name, group_var, group_def in zip(PLOT_GROUPINGS, ['njets', 'tau_dm'], ['njets', ['tau_decaymode_1', 'tau_decaymode_2']], [grouping_njets, grouping_tdm]):
                # ----- clipped FF -----
                fig_ar, ax_ar = plot_fake_factors_grouped(
                    df=df,
                    category_title=f'split in {grouping}',
                    grouping=grouping,
                    mode=args.mode,
                )
                plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / grouping / f'{args.frac}_fraction'/ f'plot_ff_splitTaus.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / grouping / f'{args.frac}_fraction'/ f'plot_ff_splitTaus.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig_ar)
                logger.info(f'Saved FF distributions in AR for {grouping}')

                # ----- clipped combined FF -----
                fig_ar_ct, ax_ar_ct = plot_fake_factors_grouped_combTaus(
                    df=df,
                    category_title=f'split in {grouping}',
                    grouping=grouping,
                    mode=args.mode,
                )
                plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / grouping / f'{args.frac}_fraction'/ f'plot_ff_combTaus.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / grouping / f'{args.frac}_fraction'/ f'plot_ff_combTaus.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig_ar_ct)
                logger.info(f'Saved FF distributions in AR for combined Taus for {grouping}')

                # ----- FF in DR -----
                fig_dr, ax_dr = plot_fake_factors_in_dr_grouped(
                    df=df,
                    category_title=f'split in {grouping}',
                    grouping=grouping,
                    mode=args.mode,
                )
                plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_DR' / grouping / f'{args.frac}_fraction'/ f'plot_ff_DR_splitTaus.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_DR' / grouping / f'{args.frac}_fraction'/ f'plot_ff_DR_splitTaus.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig_dr)
                logger.info(f'Saved FF distributions in DR for {grouping}')


                # ----- unclipped FF -----
                fig_ar, ax_ar = plot_fake_factors_grouped(
                    df=df,
                    category_title=f'split in {grouping}',
                    grouping=grouping,
                    clipped=False,
                    mode=args.mode,
                )
                plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / grouping / f'{args.frac}_fraction'/ f'plot_ff_unclipped_splitTaus.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / grouping / f'{args.frac}_fraction'/ f'plot_ff_unclipped_splitTaus.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig_ar)
                logger.info(f'Saved FF distributions in AR for {grouping}')

        # ----- combinatorial weight/ fraction factor -----
        if args.frac == 'pt_binned' and args.mode=='dnn':
            for grouping, group_name, group_var, group_def in zip(PLOT_GROUPINGS, ['njets', 'tau_dm'], ['njets', ['tau_decaymode_1', 'tau_decaymode_2']], [grouping_njets, grouping_tdm]):
                # ----- get fraction and bins -----
                cfg_frac = load_config(cfg_path['fractions']+'/fractions.yaml')
                safe_path = PLOTS_DIR / 'tau_split' / 'Fraction_factors' / grouping / f'{args.frac}_fraction' / 'pt1_pt2'/ 'tau1'
                # ----- Ar-like
                frac_arlike = cfg_frac['AR_like'][f'{group_name}']

                plot_fractions_grouped('AR_like', grouped_frac=frac_arlike, grouping=grouping, safe_path=safe_path, label=get_label)
                    
                # ----- AR
                # ----- calculate fraction in AR -----                    
                fraction_in_bins_grouped(
                    df_tau1=df.data.AR_tau1, 
                    df_tau2=df.data.AR_tau2, 
                    frac_file=cfg_path['fractions']+'/fractions.yaml', 
                    region='AR', ar_file=cfg_frac['AR_like'], 
                    grouping=group_name, grouping_variable=group_var, grouping_definition=group_def)
            
                cfg_frac = load_config(cfg_path['fractions']+'/fractions.yaml')
                frac_ar = cfg_frac['AR'][f'{group_name}']
                plot_fractions_grouped('AR', grouped_frac=frac_ar, grouping=grouping, safe_path=safe_path, label=get_label)

                logger.info(f'Saved plots of Fraction Factors for ungrouped')

                # ----- plot m_vis/pt_2
                safe_path = PLOTS_DIR / 'tau_split' / 'Fraction_factors' / grouping / f'{args.frac}_fraction' / 'mvis_pt2' / 'tau1'

                grouped_frac = fraction_in_bins_grouped(
                    df_tau1=df.data.AR_like_tau1,
                    df_tau2=df.data.AR_like_tau2,
                    frac_file=cfg_path['fractions']+'/fractions.yaml',
                    var1='m_vis', var2 = 'pt_2',
                    region='AR_like', ar_file=cfg_frac['AR_like'],
                    grouping=group_name, grouping_variable=group_var, grouping_definition=group_def
                    )

    
                plot_fractions_grouped('AR_like', grouped_frac=grouped_frac, grouping=grouping, safe_path=safe_path, label=get_label, var1='m_vis')
    
                grouped_frac = fraction_in_bins_grouped(
                    df_tau1=df.data.AR_tau1,
                    df_tau2=df.data.AR_tau2,
                    frac_file=cfg_path['fractions']+'/fractions.yaml',
                    var1='m_vis', var2 = 'pt_2',
                    region='AR', ar_file=cfg_frac['AR_like'],
                    grouping=group_name, grouping_variable=group_var, grouping_definition=group_def
                    )
    
                plot_fractions_grouped('AR', grouped_frac=grouped_frac, grouping=grouping, safe_path=safe_path, label=get_label, var1='m_vis')

                plot_in_hist1d(
                    input_values=[df.data.AR_tau1[f'fraction_pt_binned_1_{group_name}'], df.data.AR_tau2[f'fraction_pt_binned_2_{group_name}']], 
                    label=[r'$AR_1$', r'$AR_2$'],
                    cat_title=f'{group_name} inclusive',
                    input_name='combinatorial weight',
                    save_dir=PLOTS_DIR / 'tau_split' / 'Fraction_factors' / grouping / f'{args.frac}_fraction',
                    save_name='fractions',
                    )
                
        if args.frac=='DNN' and args.mode=='dnn':
            plot_in_hist1d(
                input_values=[df.data.AR_tau1['fraction_dnn_1_njets'], df.data.AR_tau2['fraction_dnn_2_njets']], 
                label=[r'$AR_1$', r'$AR_2$'],
                cat_title='njets inclusive',
                input_name='combinatorial weight',
                save_dir=PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'njets' / f'{args.frac}_fraction',
                save_name='fractions',
                )
            logger.info('Plotted fraction factors calculated with multiclass classification...')

            for x in ['0', '1', '2_1000']:
                for plot_nn_output in (plot_NN_output_frac_stacked, plot_NN_output_frac_overlay, plot_NN_output_frac_ratio, plot_NN_output_frac_events):
                    plot_nn_output(
                        df=df,
                        cat_title=f'njets = {x}' if x != '2_1000' else r'$2 \leq n_{jets} \leq 1000$',
                        model_path=cfg_path['multiclass_results'] + f'/split/njets/{x}',
                        n_classes=2,
                        training_variables=training_variables_multiclass,
                        grouping='njets',
                        group_key=x,
                        grouping_definition=grouping_njets,
                        save_dir=PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'njets' / f'{args.frac}_fraction',
                        save_name=f'nn_output_{x}',
                        wo_train=args.wo_train
                    )
            for plot_nn_output in (plot_NN_output_frac_stacked, plot_NN_output_frac_overlay, plot_NN_output_frac_ratio, plot_NN_output_frac_events):
                plot_nn_output(
                    df=df,
                    cat_title='tau decaymode inclusive',
                    model_path=cfg_path['multiclass_results'] + '/split/tau_dm/combined',
                    n_classes=2,
                    training_variables=training_variables_multiclass,
                    grouping='tau_dm',
                    group_key='combined',
                    save_dir=PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'tau_decaymode' / f'{args.frac}_fraction',
                    save_name='nn_output',
                    wo_train=args.wo_train,
                )
            logger.info('Plotted NN outputs of multiclass classification...')
        
        # ----- FF closure in AR -----
        if args.closure_AR:
            x = uproot.open("/work/ptoedter/MA-Pascal/smhtt_ul/output/2018-et-2025-12_15_with_uncertainties_ntupels_v1-final_v3_2026_02_19/control_shapes-2018-et-2025-12_15_with_uncertainties_ntupels_v1-final_v3_2026_02_19.root")
            bkgs = [it for it in x.keys() if "#q_1;" in it and "Nominal" in it and any(subit in it for subit in ["TT-TTL", "DY-ZL", "jetFakes#", "VV-VVL", "EMB#"])]
            corr_emb_ff = sum([x[it].to_numpy()[0] for it in bkgs]) / [x[next(it for it in x.keys() if "data" in it and "Nominal" in it and "#q_1" in it)].to_numpy()[0]]
    
            
            for var in variables:
                bins, label = get_bins_and_label(var)
                label = labels_cfg['tt'][var]
                fig, ax, _ = plot_closure(
                        df = df,
                        var = var,
                        bins = bins,
                        label = label,
                        grouping = 'tau_decaymode',
                        AR_comb=args.AR_comb,
                        mode=args.mode,
                    )
                
                plt.savefig(PLOTS_DIR / 'tau_split' / 'control_plots' / 'tau_decaymode' / f'{args.frac}_fraction' / f'plot_closure_{var}.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_split' / 'control_plots' / 'tau_decaymode' / f'{args.frac}_fraction' / f'plot_closure_{var}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)

            logger.info('Saved all closure plots in tau decaymode')


            for var in variables:
                bins, label = get_bins_and_label(var)
                label = labels_cfg['tt'][var]
                fig, ax, _ = plot_closure(
                        df = df,
                        var = var,
                        bins = bins,
                        label = label,
                        grouping = 'njets',
                        AR_comb = args.AR_comb,
                        mode = args.mode
                    )
                
                plt.savefig(PLOTS_DIR / 'tau_split' / 'control_plots' / 'njets' / f'{args.frac}_fraction' / f'plot_closure_{var}.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_split' / 'control_plots' / 'njets' / f'{args.frac}_fraction' / f'plot_closure_{var}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)

            logger.info('Saved all closure plots in njets')
        
    elif args.taus=='split' and not args.dnn_grouped:
        logger.info('Initiaize plotting for tau split FF calculated through single DNN...')

        df = load_data(DATA_PATH, MASKS_PATH)

        # ----- Closure plots in DR -----
        if args.closure_DR:
            for var in variables:
                bins, label = get_bins_and_label(var)
                label = labels_cfg['tt'][var]

                fig_q, _ = FF_closure_in_DR_tau1(
                    df=df,
                    var=var,
                    bins=bins,
                    label=label,
                    grouping=None,
                )
                plt.savefig(PLOTS_DIR / 'tau_split' / 'closure_in_DR' / 'ungrouped' / f'{args.frac}_fraction' / f'FF_closure_DR_tau1_{var}.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_split' / 'closure_in_DR' / 'ungrouped' / f'{args.frac}_fraction' / f'FF_closure_DR_tau1_{var}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig_q)

                fig_q, _ = FF_closure_in_DR_tau2(
                    df=df,
                    var=var,
                    bins=bins,
                    label=label,
                    grouping=None,
                )
                plt.savefig(PLOTS_DIR / 'tau_split' / 'closure_in_DR' / 'ungrouped' / f'{args.frac}_fraction' / f'FF_closure_DR_tau2_{var}.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_split' / 'closure_in_DR' / 'ungrouped' / f'{args.frac}_fraction' / f'FF_closure_DR_tau2_{var}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig_q)

            logger.info(f'Saved closure plots in DR for ungrouped')

        # ----- Fake-factor distributions -----
        if args.FF_dist:          

            fig_ar, ax_ar = plot_fake_factors(df=df)
            plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_ff_splitTaus.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_ff_splitTaus.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved FF distributions in AR for ungrouped DNN')

            fig_ar, ax_ar = plot_fake_factors(df=df, clipped=False)
            plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_ff_unclipped_splitTaus.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_ff_unclipped_splitTaus.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved unclipped FF distributions in AR for ungrouped DNN')

            # ----- clipped combined FF -----
            fig_ar_ct, ax_ar_ct = plot_fake_factors_combTaus(df=df)
            plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_ff_combTaus.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_split' / 'FF_distribution_AR' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_ff_combTaus.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar_ct)
            logger.info(f'Saved FF distributions in AR for combined Taus for ungrouped')

        if args.frac == 'pt_binned':
            # ----- get fraction and bins -----
            cfg_frac = load_config(cfg_path['fractions']+'/fractions.yaml')

            # ----- Ar-like
            frac_arlike = cfg_frac['AR_like']['ungrouped']
            frac, pt1_edges, pt2_edges = frac_arlike['fraction'], frac_arlike['pt1_edges'], frac_arlike['pt2_edges']
            mean, std = frac_arlike['global_frac'], frac_arlike['global_std']
            
            fig, ax = plot_fractions('AR_like', frac=frac, var1_edges=pt1_edges, var2_edges=pt2_edges, global_frac=mean, global_std=std, label=get_label)
            plt.savefig(PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'pt1_pt2' / 'tau1' / 'plot_fractions_ARlike.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'pt1_pt2' / 'tau1' / 'plot_fractions_ARlike.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig)

            # ----- AR
            # ----- calculate fraction in AR -----
            fraction_in_bins(
                df_tau1=df.data.AR_tau1,
                df_tau2=df.data.AR_tau2,
                frac_file=cfg_path['fractions']+'/fractions.yaml',
                region='AR', var1_bin_edges=pt1_edges, var2_bin_edges=pt2_edges
                )

            cfg_frac = load_config(cfg_path['fractions']+'/fractions.yaml')
            frac_ar = cfg_frac['AR']['ungrouped']
            fraction_ar, pt1_edges_ar, pt2_edges_ar = frac_ar['fraction'], frac_ar['pt1_edges'], frac_ar['pt2_edges']
            mean_ar, std_ar = frac_ar['global_frac'], frac_ar['global_std']

            fig, ax = plot_fractions('AR', frac=fraction_ar, var1_edges=pt1_edges_ar, var2_edges=pt2_edges_ar, global_frac=mean_ar, global_std=std_ar, label=get_label)
            plt.savefig(PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'pt1_pt2' / 'tau1' / 'plot_fractions_AR.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'pt1_pt2' / 'tau1' / 'plot_fractions_AR.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig)

            # ----- plot diff -----
            frac_diff = np.array(frac) - np.array(fraction_ar)
            h = frac_diff.flatten()
            h = h[~np.isnan(h)]
            mean_diff, std_diff = np.mean(h), np.std(h)

            fig, ax = plot_fractions('AR_like - AR', frac=frac_diff, var1_edges=pt1_edges_ar, var2_edges=pt2_edges_ar, global_frac=mean_diff, global_std=std_diff, label=get_label)
            plt.savefig(PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'pt1_pt2' / 'tau1' / 'plot_fractions_diff.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'pt1_pt2' / 'tau1' / 'plot_fractions_diff.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f'Saved plots of Fraction Factors for ungrouped')

            # ----- plot m_vis/pt_2
            frac, edges1, edges2, mean, std = fraction_in_bins(
                df_tau1=df.data.AR_like_tau1,
                df_tau2=df.data.AR_like_tau2,
                frac_file=cfg_path['fractions']+'/fractions.yaml',
                var1='m_vis', var2 = 'pt_2',
                region='AR_like', var1_bin_edges=pt1_edges, var2_bin_edges=pt2_edges
                )

            fig, ax = plot_fractions('AR_like', frac=frac, var1_edges=edges1, var2_edges=edges2, global_frac=mean, global_std=std, label=get_label, var1='m_vis')
            plt.savefig(PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'mvis_pt2' / 'tau1' / 'plot_fractions_AR_like.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'mvis_pt2' / 'tau1' / 'plot_fractions_AR_like.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig)

            frac, edges1, edges2, mean, std = fraction_in_bins(
                df_tau1=df.data.AR_tau1,
                df_tau2=df.data.AR_tau2,
                frac_file=cfg_path['fractions']+'/fractions.yaml',
                var1='m_vis', var2 = 'pt_2',
                region='AR', var1_bin_edges=pt1_edges, var2_bin_edges=pt2_edges
                )

            fig, ax = plot_fractions('AR', frac=frac, var1_edges=edges1, var2_edges=edges2, global_frac=mean, global_std=std, label=get_label, var1='m_vis')
            plt.savefig(PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'mvis_pt2' / 'tau1' / 'plot_fractions_AR.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'mvis_pt2' / 'tau1' / 'plot_fractions_AR.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig)

            plot_in_hist1d(
                input_values=[df.data.AR_tau1['fraction_pt_binned_1'], df.data.AR_tau2['fraction_pt_binned_2']], 
                label=[r'$AR_1$', r'$AR_2$'],
                cat_title='inclusive',
                input_name='combinatorial weight',
                save_dir=PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction',
                save_name='fractions',
                )

        elif args.frac == 'DNN':
            plot_in_hist1d(
                input_values=[df.data.AR_tau1['fraction_dnn_1'], df.data.AR_tau2['fraction_dnn_2']], 
                label=[r'$AR_1$', r'$AR_2$'],
                cat_title='inclusive',
                input_name='combinatorial weight',
                save_dir=PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction',
                save_name='fractions',
                )
            logger.info('Plotted fraction factors calculated with multiclass classification...')

            for plot_nn_output in (plot_NN_output_frac_stacked, plot_NN_output_frac_overlay, plot_NN_output_frac_ratio, plot_NN_output_frac_events):
                plot_nn_output(
                    df = df,
                    cat_title='inclusive',
                    model_path=cfg_path['multiclass_results']+'/split/ungrouped',
                    n_classes=2,
                    training_variables=training_variables_multiclass,
                    save_dir=PLOTS_DIR / 'tau_split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction',
                    save_name='nn_output',
                    wo_train=args.wo_train,
                )
            logger.info('Plotted NN outputs of multiclass classification...')
        
        # ----- FF closure in AR -----
        if args.closure_AR:
            for var in variables:
                bins, label = get_bins_and_label(var)
                label = labels_cfg['tt'][var]
                fig, ax, _ = plot_closure(
                        df = df,
                        var = var,
                        bins = bins,
                        label = label,
                        grouping = None,
                        AR_comb = args.AR_comb
                    )
                
                plt.savefig(PLOTS_DIR / 'tau_split' / 'control_plots' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_closure_{var}.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_split' / 'control_plots' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_closure_{var}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)

            logger.info('Saved all closure plots for ungrouped DNN')
        

    # ----- tau 3-split FF -----

    elif args.taus=='3split' and not args.dnn_grouped:
        logger.info('Initiaize plotting for tau 3-split FF calculated through single DNN...')

        df = load_data(DATA_PATH, MASKS_PATH_3SPLIT)
        if args.AR_comb:
            ar1 = ar2 = ar3 = df.data.AR_comb
        else:
            ar1, ar2, ar3 = df.data.AR_1, df.data.AR_2, df.data.AR_3
        # ----- Closure plots in DR -----
        if args.closure_DR:
            for num, split in zip(['1', '2', '3'], ['tau1', 'tau2', 'tau1&2']):
                if split == 'tau1': df_arlike = df.data.AR_like_1
                elif split == 'tau2': df_arlike = df.data.AR_like_2
                elif split == 'tau1&2': df_arlike = df.data.AR_like_3                           
            
                for var in variables:
                    bins, label = get_bins_and_label(var)
                    label = labels_cfg['tt'][var]
    
                    fig_q, _ = FF_closure_in_DR_3split(
                        df_srlike=df.data.SR_like,
                        df_arlike=df_arlike,
                        var=var,
                        bins=bins,
                        label=label,
                        split=num,
                        grouping=None,
                    )
                    plt.savefig(PLOTS_DIR / 'tau_3split' / 'closure_in_DR' / 'ungrouped' / f'{args.frac}_fraction' / f'FF_closure_DR_{split}_{var}.png', dpi=150, bbox_inches='tight')
                    plt.savefig(PLOTS_DIR / 'tau_3split' / 'closure_in_DR' / 'ungrouped' / f'{args.frac}_fraction' / f'FF_closure_DR_{split}_{var}.pdf', dpi=150, bbox_inches='tight')
                    plt.close(fig_q)
    
            logger.info(f'Saved closure plots in DR for ungrouped')

        # ----- Fake-factor distributions -----
        if args.FF_dist:          

            fig_ar, ax_ar = plot_fake_factors_3split(df=df, category_title='inclusive', AR_comb=args.AR_comb)
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_ff_3splitTaus.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_ff_3splitTaus.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved FF distributions in AR for ungrouped DNN')

            fig_ar, ax_ar = plot_fake_factors_3split(df=df, clipped=False, category_title='inclusive', AR_comb=args.AR_comb)
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_ff_unclipped_3splitTaus.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_ff_unclipped_3splitTaus.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved unclipped FF distributions in AR for ungrouped DNN')

            fig_ar, ax_ar = plot_fake_factors_3split(df=df, category_title='inclusive', in_one_plot=True, AR_comb=args.AR_comb)
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_ff_3splitTaus_allin1.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_ff_3splitTaus_allin1.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved FF distributions in AR for ungrouped DNN')
            # ----- clipped combined FF -----
            fig_ar_ct, ax_ar_ct = plot_fake_factors_combTaus_3split(df=df, AR_comb=args.AR_comb)
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_ff_combTaus.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_ff_combTaus.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar_ct)
            logger.info(f'Saved FF distributions in AR for combined Taus for ungrouped')

        if args.frac == 'pt_binned':
            for tau in ['tau1', 'tau2', 'tau1&2']:

                # ----- get fraction and bins -----
                cfg_frac = load_config(cfg_path['fractions']+f'/fractions_{tau}.yaml')
                # ----- Ar-like
                frac_arlike = cfg_frac['AR_like']['ungrouped']
                frac, pt1_edges, pt2_edges = frac_arlike['fraction'], frac_arlike['pt1_edges'], frac_arlike['pt2_edges']
                mean, std = frac_arlike['global_frac'], frac_arlike['global_std']
                
                fig, ax = plot_fractions_3split('AR_like', frac=frac, var1_edges=pt1_edges, var2_edges=pt2_edges, global_frac=mean, global_std=std, label=get_label, max_bins=args.fraction_max_bins)
                plt.savefig(PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'pt1_pt2' / tau / f'plot_fractions_ARlike.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'pt1_pt2' / tau / f'plot_fractions_ARlike.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)

                # ----- AR
                # ----- calculate fraction in AR -----
                fraction_in_bins_3split(
                    which_frac=tau, 
                    df_tau1=df.data.AR_1, 
                    df_tau2=df.data.AR_2, 
                    df_tau3=df.data.AR_3, 
                    frac_file=cfg_path['fractions']+f'/fractions_{tau}.yaml',
                    region='AR', var1_bin_edges=pt1_edges, var2_bin_edges=pt2_edges)

                cfg_frac = load_config(cfg_path['fractions']+f'/fractions_{tau}.yaml')
                frac_ar = cfg_frac['AR']['ungrouped']
                fraction_ar, pt1_edges_ar, pt2_edges_ar = frac_ar['fraction'], frac_ar['pt1_edges'], frac_ar['pt2_edges']
                mean_ar, std_ar = frac_ar['global_frac'], frac_ar['global_std']

                fig, ax = plot_fractions_3split('AR', frac=fraction_ar, var1_edges=pt1_edges_ar, var2_edges=pt2_edges_ar, global_frac=mean_ar, global_std=std_ar, label=get_label, max_bins=args.fraction_max_bins)
                plt.savefig(PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'pt1_pt2' / tau / f'plot_fractions_AR.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'pt1_pt2' / tau / f'plot_fractions_AR.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)

                # ----- plot diff -----
                frac_diff = np.array(frac) - np.array(fraction_ar)
                h = frac_diff.flatten()
                h = h[~np.isnan(h)]
                mean_diff, std_diff = np.mean(h), np.std(h)

                fig, ax = plot_fractions_3split('AR_like - AR', frac=frac_diff, var1_edges=pt1_edges_ar, var2_edges=pt2_edges_ar, global_frac=mean_diff, global_std=std_diff, label=get_label, max_bins=args.fraction_max_bins)
                plt.savefig(PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'pt1_pt2' / tau / f'plot_fractions_diff.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'pt1_pt2' / tau / f'plot_fractions_diff.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)

                logger.info(f'Saved plots of Fraction Factors for ungrouped')

                # ----- plot m_vis/pt_2
                frac, edges1, edges2, mean, std = fraction_in_bins_3split(
                    which_frac=tau,
                    df_tau1=df.data.AR_like_1,
                    df_tau2=df.data.AR_like_2,
                    df_tau3=df.data.AR_like_3,
                    frac_file=cfg_path['fractions']+'/fractions.yaml',
                    var1='m_vis', var2 = 'pt_2',
                    region='AR_like', var1_bin_edges=pt1_edges, var2_bin_edges=pt2_edges
                    )
    
                fig, ax = plot_fractions_3split('AR_like', frac=frac, var1_edges=edges1, var2_edges=edges2, global_frac=mean, global_std=std, label=get_label, var1='m_vis')
                plt.savefig(PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'mvis_pt2' / f'{tau}' / 'plot_fractions_AR_like.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'mvis_pt2' / f'{tau}' / 'plot_fractions_AR_like.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)

                frac, edges1, edges2, mean, std = fraction_in_bins_3split(
                    which_frac=tau,
                    df_tau1=df.data.AR_1,
                    df_tau2=df.data.AR_2,
                    df_tau3=df.data.AR_3,
                    frac_file=cfg_path['fractions']+'/fractions.yaml',
                    var1='m_vis', var2 = 'pt_2',
                    region='AR', var1_bin_edges=pt1_edges, var2_bin_edges=pt2_edges
                    )
    
                fig, ax = plot_fractions_3split('AR_like', frac=frac, var1_edges=edges1, var2_edges=edges2, global_frac=mean, global_std=std, label=get_label, var1='m_vis')
                plt.savefig(PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'mvis_pt2' / f'{tau}' / 'plot_fractions_AR.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction' / 'mvis_pt2' / f'{tau}' / 'plot_fractions_AR.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)

                logger.info(f'Plotted fraction factor in bins of pt_2 and m_vis for {tau}')

            plot_in_hist1d(
                input_values=[ar1['fraction_pt_binned_1'], ar2['fraction_pt_binned_2'], ar3['fraction_pt_binned_3']],
                label=[r'$AR_1$', r'$AR_2$', r'$AR_3$'],
                cat_title='inclusive',
                input_name='combinatorial weight',
                save_dir=PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction',
                save_name='fractions',
                )
            
        elif args.frac == 'DNN':
            plot_in_hist1d(
                input_values=[ar1['fraction_dnn_1'], ar2['fraction_dnn_2'], ar3['fraction_dnn_3']],
                label=[r'$AR_1$', r'$AR_2$', r'$AR_3$'],
                cat_title='inclusive',
                input_name='combinatorial weight',
                save_dir=PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction',
                save_name='fractions',
                )
            logger.info('Plotted fraction factors calculated with multiclass classification...')

            for plot_nn_output in (plot_NN_output_frac_stacked, plot_NN_output_frac_overlay, plot_NN_output_frac_ratio, plot_NN_output_frac_events):
                plot_nn_output(
                    df = df,
                    cat_title='inclusive',
                    model_path=cfg_path['multiclass_results']+'/3split/ungrouped',
                    n_classes=3,
                    training_variables=training_variables_multiclass,
                    save_dir=PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / 'ungrouped' / f'{args.frac}_fraction',
                    save_name='nn_output',
                    wo_train=args.wo_train,
                )
            logger.info('Plotted NN outputs of multiclass classification...')





        # ----- FF closure in AR -----
        if args.closure_AR:
                for var in variables:
                    bins, label = get_bins_and_label(var)
                    label = labels_cfg['tt'][var]
                    fig, ax, _ = plot_closure_3split(
                            df = df,
                            var = var,
                            bins = bins,
                            label = label,
                            grouping = None,
                            AR_comb=args.AR_comb,
                        )
                    
                    plt.savefig(PLOTS_DIR / 'tau_3split' / 'control_plots' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_closure_{var}.png', dpi=150, bbox_inches='tight')
                    plt.savefig(PLOTS_DIR / 'tau_3split' / 'control_plots' / 'ungrouped' / f'{args.frac}_fraction' / f'plot_closure_{var}.pdf', dpi=150, bbox_inches='tight')
                    plt.close(fig)
    
                logger.info('Saved all closure plots for ungrouped DNN')

    elif args.taus=='3split' and args.dnn_grouped:
        logger.info('Initiaize plotting for tau 3split FF calculated through grouped DNN...')

        df = load_data(DATA_PATH, MASKS_PATH_3SPLIT)
        if args.AR_comb:
            ar1 = ar2 = ar3 = df.data.AR_comb
        else:
            ar1, ar2, ar3 = df.data.AR_1, df.data.AR_2, df.data.AR_3
        grouping = 'njets'
        # ----- Closure plots in DR -----
        if args.closure_DR:
            for num, split in zip(['1', '2', '3'], ['tau1', 'tau2', 'tau1&2']):
                if split == 'tau1': df_arlike = df.data.AR_like_1
                elif split == 'tau2': df_arlike = df.data.AR_like_2
                elif split == 'tau1&2': df_arlike = df.data.AR_like_3

                for var in variables:
                    bins, label = get_bins_and_label(var)
                    label = labels_cfg['tt'][var]

                    fig_q, _ = FF_closure_in_DR_3split(
                        df_srlike=df.data.SR_like,
                        df_arlike=df_arlike,
                        var=var,
                        bins=bins,
                        label=label,
                        split=num,
                        grouping=grouping,
                    )
                    plt.savefig(PLOTS_DIR / 'tau_3split' / 'closure_in_DR' / grouping / f'{args.frac}_fraction'/ f'FF_closure_DR_{split}_{var}.png', dpi=150, bbox_inches='tight')
                    plt.savefig(PLOTS_DIR / 'tau_3split' / 'closure_in_DR' / grouping / f'{args.frac}_fraction'/ f'FF_closure_DR_{split}_{var}.pdf', dpi=150, bbox_inches='tight')
                    plt.close(fig_q)

                logger.info(f'Saved closure plots in DR for {grouping}')

        # ----- Fake-factor distributions -----
        if args.FF_dist:
            
            # ----- clipped FF -----
            fig_ar, ax_ar = plot_fake_factors_grouped_3split(df=df, category_title=f'split in {grouping}', grouping=grouping, AR_comb=args.AR_comb)
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / grouping / f'{args.frac}_fraction'/ f'plot_ff_3split.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / grouping / f'{args.frac}_fraction'/ f'plot_ff_3split.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved FF distributions in AR for {grouping}')

            fig_ar, ax_ar = plot_fake_factors_3split(df=df, category_title='inclusive', grouping=grouping, in_one_plot=True, AR_comb=args.AR_comb)
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / grouping / f'{args.frac}_fraction' / f'plot_ff_3splitTaus_allin1.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / grouping / f'{args.frac}_fraction' / f'plot_ff_3splitTaus_allin1.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)

            # ----- clipped combined FF -----
            fig_ar_ct, ax_ar_ct = plot_fake_factors_grouped_combTaus_3split(df=df, category_title=f'split in {grouping}', grouping=grouping, AR_comb=args.AR_comb)
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / grouping / f'{args.frac}_fraction'/ f'plot_ff_combTaus.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / grouping / f'{args.frac}_fraction'/ f'plot_ff_combTaus.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar_ct)
            logger.info(f'Saved FF distributions in AR for combined Taus for {grouping}')

            # ----- unclipped FF -----
            fig_ar, ax_ar = plot_fake_factors_grouped_3split(df=df, category_title=f'split in {grouping}', grouping=grouping, clipped=False, AR_comb=args.AR_comb)
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / grouping / f'{args.frac}_fraction'/ f'plot_ff_unclipped_3split.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'tau_3split' / 'FF_distribution_AR' / grouping / f'{args.frac}_fraction'/ f'plot_ff_unclipped_3split.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved FF distributions in AR for {grouping}')

        # ----- plot combinatoral weight/fraction factor -----
        if args.frac == 'pt_binned':
            # ----- get fraction and bins -----
            for tau in ['tau1', 'tau2', 'tau1&2']:
                cfg_frac = load_config(cfg_path['fractions']+f'/fractions_{tau}.yaml')
                safe_path = PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / grouping / f'{args.frac}_fraction' / 'pt1_pt2' / tau
                # ----- Ar-like
                frac_arlike = cfg_frac['AR_like'][f'njets']

                plot_fractions_grouped_3split('AR_like', grouped_frac=frac_arlike, grouping=grouping, safe_path=safe_path, tau=tau)
                    
                # ----- AR
                # ----- calculate fraction in AR -----                    
                fraction_in_bins_grouped_3split(
                    which_frac=tau, 
                    df_tau1=df.data.AR_1, 
                    df_tau2=df.data.AR_2, 
                    df_tau3=df.data.AR_3, 
                    frac_file=cfg_path['fractions']+f'/fractions_{tau}.yaml', 
                    region='AR', ar_file=cfg_frac['AR_like'], grouping='njets', grouping_variable='njets', grouping_definition=grouping_njets)
            
                cfg_frac = load_config(cfg_path['fractions']+f'/fractions_{tau}.yaml')
                frac_ar = cfg_frac['AR'][f'njets']
                plot_fractions_grouped_3split('AR', grouped_frac=frac_ar, grouping=grouping, safe_path=safe_path, tau=tau)

                logger.info(f'Saved plots of Fraction Factors for njets')

                # ----- plot m_vis/pt_2
                grouped_frac = fraction_in_bins_grouped_3split(
                    which_frac=tau,
                    df_tau1=df.data.AR_like_1,
                    df_tau2=df.data.AR_like_2,
                    df_tau3=df.data.AR_like_3,
                    frac_file=cfg_path['fractions']+f'/fractions_{tau}.yaml',
                    var1='m_vis', var2 = 'pt_2',
                    region='AR_like', ar_file=cfg_frac['AR_like'], 
                    grouping='njets', grouping_variable='njets', grouping_definition=grouping_njets
                    )

                safe_path = PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / grouping / f'{args.frac}_fraction' / 'mvis_pt2' / f'{tau}'
                plot_fractions_grouped_3split('AR_like', grouped_frac=grouped_frac, grouping=grouping, safe_path=safe_path, tau=tau, label=get_label, var1='m_vis')
                

                grouped_frac = fraction_in_bins_grouped_3split(
                    which_frac=tau,
                    df_tau1=df.data.AR_1,
                    df_tau2=df.data.AR_2,
                    df_tau3=df.data.AR_3,
                    frac_file=cfg_path['fractions']+f'/fractions_{tau}.yaml',
                    var1='m_vis', var2 = 'pt_2',
                    region='AR', ar_file=cfg_frac['AR_like'], 
                    grouping='njets', grouping_variable='njets', grouping_definition=grouping_njets
                    )
    
                safe_path = PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / grouping / f'{args.frac}_fraction' / 'mvis_pt2' / f'{tau}'
                plot_fractions_grouped_3split('AR', grouped_frac=grouped_frac, grouping=grouping, safe_path=safe_path, tau=tau, label=get_label, var1='m_vis')

                logger.info(f'Plotted fraction factor in bins of pt_2 and m_vis for {tau}')

            plot_in_hist1d(
                input_values=[ar1['fraction_pt_binned_1_njets'], ar2['fraction_pt_binned_2_njets'], ar3['fraction_pt_binned_3_njets']],
                label=[r'$AR_1$', r'$AR_2$', r'$AR_3$'],
                cat_title='njets inclusive',
                input_name='combinatorial weight',
                save_dir=PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / grouping / f'{args.frac}_fraction',
                save_name='fractions',
                )
            logger.info('Plotted fraction factors calculated with multiclass classification...')
                
        if args.frac=='DNN':
            plot_in_hist1d(
                input_values=[ar1['fraction_dnn_1_njets'], ar2['fraction_dnn_2_njets'], ar3['fraction_dnn_3_njets']],
                label=[r'$AR_1$', r'$AR_2$', r'$AR_3$'],
                cat_title='njets inclusive',
                input_name='combinatorial weight',
                save_dir=PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / grouping / f'{args.frac}_fraction',
                save_name='fractions',
                )
            logger.info('Plotted fraction factors calculated with multiclass classification...')

            for x in ['0', '1', '2_1000']:
                for plot_nn_output in (plot_NN_output_frac_stacked, plot_NN_output_frac_overlay, plot_NN_output_frac_ratio, plot_NN_output_frac_events):
                    plot_nn_output(
                        df = df,
                        cat_title=f'njets = {x}' if x != '2_1000' else r'$2 \leq n_{jets} \leq 1000$',
                        model_path=cfg_path['multiclass_results']+f'/3split/njets/{x}',
                        n_classes=3,
                        training_variables=training_variables_multiclass,
                        grouping='njets',
                        group_key=x,
                        grouping_definition=grouping_njets,
                        save_dir=PLOTS_DIR / 'tau_3split' / 'Fraction_factors' / 'njets' / f'{args.frac}_fraction',
                        save_name=f'nn_output_{x}',
                        wo_train=args.wo_train,
                    )
            logger.info('Plotted NN outputs of multiclass classification...')


        # ----- FF closure in AR -----
        if args.closure_AR:
            #x = uproot.open("/work/ptoedter/MA-Pascal/smhtt_ul/output/2018-et-2025-12_15_with_uncertainties_ntupels_v1-final_v3_2026_02_19/control_shapes-2018-et-2025-12_15_with_uncertainties_ntupels_v1-final_v3_2026_02_19.root")
            #bkgs = [it for it in x.keys() if "#q_1;" in it and "Nominal" in it and any(subit in it for subit in ["TT-TTL", "DY-ZL", "jetFakes#", "VV-VVL", "EMB#"])]
            #corr_emb_ff = sum([x[it].to_numpy()[0] for it in bkgs]) / [x[next(it for it in x.keys() if "data" in it and "Nominal" in it and "#q_1" in it)].to_numpy()[0]]

            for var in variables:
                bins, label = get_bins_and_label(var)
                label = labels_cfg['tt'][var]
                fig, ax, _ = plot_closure_3split(
                        df = df,
                        var = var,
                        bins = bins,
                        label = label,
                        grouping = 'njets',
                        AR_comb=args.AR_comb,
                    )
                
                plt.savefig(PLOTS_DIR / 'tau_3split' / 'control_plots' / 'njets' / f'{args.frac}_fraction' / f'plot_closure_{var}.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / 'tau_3split' / 'control_plots' / 'njets' / f'{args.frac}_fraction' / f'plot_closure_{var}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)

            logger.info('Saved all closure plots in njets')



   # ----- tau inclusive FF -----

    elif args.taus=='incl' and args.dnn_grouped:
        logger.info('Initiaize plotting for tau inclusive FF calculated through grouped DNN...')

        if args.incl=='and': incl = 0
        elif args.incl=='or': incl = 1
        elif args.incl=='andor': incl = 2
        else:
            logger.error(f'Value Error: args.incl = {args.incl}, but only accepts "and or "or".')
            exit()
        df = load_data(DATA_PATH, MASKS_PATH_INCL[incl])

        # ----- Closure plots in DR -----
        if args.closure_DR:
            for var in variables:
                bins, label = get_bins_and_label(var)
                label = labels_cfg['tt'][var]

                fig_q, _ = FF_closure_in_DR_incl(
                    df=df,
                    incl=args.incl,
                    var=var,
                    bins=bins,
                    label=label,
                    grouping='njets',
                )
                plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'closure_in_DR' / 'njets' / f'FF_closure_DR_incl_{args.incl}_{var}.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'closure_in_DR' / 'njets' / f'FF_closure_DR_incl_{args.incl}_{var}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig_q)

            logger.info(f'Saved closure plots for tau inclusive in DR for njets.')
            
        # ----- Fake-factor distributions -----
        if args.FF_dist:
            # ----- DNN FF -----
            fig_ar, ax_ar = plot_fake_factors_grouped_incl(df=df, incl=args.incl, category_title=r'split in $N_{jets}$', grouping='njets')
            plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'FF_distribution_AR' / 'njets' / f'plot_ff_incl_{args.incl}.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'FF_distribution_AR' / 'njets' / f'plot_ff_incl_{args.incl}.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved inclusive FF distributions in AR for grouped DNN')

            fig_ar, ax_ar = plot_fake_factors_grouped_incl(df=df, incl=args.incl, clipped=False, category_title=r'split in $N_{jets}$', grouping='njets')
            plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'FF_distribution_AR' / 'njets' / f'plot_ff_unclipped_incl_{args.incl}.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'FF_distribution_AR' / 'njets' / f'plot_ff_unclipped_incl_{args.incl}.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved inclusive unclipped FF distributions in AR for grouped DNN')


        # ----- FF closure in AR -----
        if args.closure_AR:
            for var in variables:
                bins, label = get_bins_and_label(var)
                label = labels_cfg['tt'][var]
                fig, ax, _ = plot_closure_incl(
                        df = df,
                        incl = args.incl,
                        var = var,
                        bins = bins,
                        label = label,
                        grouping = 'njets'
                    )
                
                plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'control_plots' / 'njets' / f'plot_closure_{var}.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'control_plots' / 'njets' / f'plot_closure_{var}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)

            logger.info('Saved all closure plots for tau inclusive grouped DNN')

    elif args.taus=='incl' and not args.dnn_grouped:
        logger.info('Initiaize plotting for tau inclusive FF calculated through single DNN...')

        if args.incl=='and': incl = 0
        elif args.incl=='or': incl = 1
        elif args.incl=='andor': incl = 2
        else:
            logger.error(f'Value Error: args.incl = {args.incl}, but only accepts "and or "or".')
            exit()

        df = load_data(DATA_PATH, MASKS_PATH_INCL[incl])
        print(len(df.data.AR))

        # ----- Closure plots in DR -----
        if args.closure_DR:
            for var in variables:
                bins, label = get_bins_and_label(var)
                label = labels_cfg['tt'][var]

                fig_q, _ = FF_closure_in_DR_incl(
                    df=df,
                    incl=args.incl,
                    var=var,
                    bins=bins,
                    label=label,
                    grouping=None,
                )
                plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'closure_in_DR' / 'ungrouped' / f'FF_closure_DR_incl_{args.incl}_{var}.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'closure_in_DR' / 'ungrouped' / f'FF_closure_DR_incl_{args.incl}_{var}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig_q)

            logger.info(f'Saved closure plots for tau inclusive in DR for ungrouped DNN.')

        # ----- Fake-factor distributions -----
        if args.FF_dist:
            # ----- DNN FF -----
            fig_ar, ax_ar = plot_fake_factors_incl(df=df, incl=args.incl)
            plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'FF_distribution_AR' / 'ungrouped' / f'plot_ff_incl_{args.incl}.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'FF_distribution_AR' / 'ungrouped' / f'plot_ff_incl_{args.incl}.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved inclusive FF distributions in AR for ungrouped DNN')

            fig_ar, ax_ar = plot_fake_factors_incl(df=df, incl=args.incl, clipped=False)
            plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'FF_distribution_AR' / 'ungrouped' / f'plot_ff_unclipped_incl_{args.incl}.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'FF_distribution_AR' / 'ungrouped' / f'plot_ff_unclipped_incl_{args.incl}.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_ar)
            logger.info(f'Saved inclusive unclipped FF distributions in AR for ungrouped DNN')

        # ----- FF closure in AR -----
        if args.closure_AR:
            for var in variables:
                bins, label = get_bins_and_label(var)
                label = labels_cfg['tt'][var]
                fig, ax, _ = plot_closure_incl(
                        df = df,
                        incl = args.incl,
                        var = var,
                        bins = bins,
                        label = label,
                        grouping = None
                    )
                
                plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'control_plots' / 'ungrouped' / f'plot_closure_{var}.png', dpi=150, bbox_inches='tight')
                plt.savefig(PLOTS_DIR / f'tau_incl_{args.incl}' / 'control_plots' / 'ungrouped' / f'plot_closure_{var}.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)

            logger.info('Saved all closure plots for tau inclusive ungrouped DNN')


# -------------------------------------------

if __name__ == '__main__':
    main()
