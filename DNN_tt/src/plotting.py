from classes import load_variables, load_data, load_model, load_fold_combined_model, test_data
from classes import calculate_fake_factors, calculate_fake_factor_dnn, calculate_fake_factor_classic, calculate_fake_factors_in_DR_wjets, calculate_fake_factors_in_DR_qcd
from classes import FF_closure_in_DR_wjets, FF_closure_in_DR_qcd, plot_fake_factors_in_DR, plot_fake_factors
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch as t
import pandas as pd
import correctionlib as cr
from classes import CMS_CHANNEL_TITLE, CMS_CATEGORY_TITLE, CMS_LUMI_TITLE, CMS_LABEL, adjust_ylim_for_legend, plot_closure, plot_fake_factors_grouped, plot_fake_factors_in_dr_grouped
from pathlib import Path
import matplotlib
import yaml
import uproot

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


DATA_PATH = '../../data/data_complete.feather'
MASKS_PATH = '../configs/masks.yaml'
TRAINING_VAR_PATH = '../configs/training_variables.yaml'
NN_CONFIG_PATH = '../configs/DNN.yaml'
CHECKPOINT_DIR = '../Training_results'

PLOTTING_CONFIG_PATH = '../configs/plotting.yaml'
LABELS_CONFIG_PATH = '../configs/labels.yaml'

PLOTS_DIR = Path('../plots/layers_3/ReLU')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_GROUPINGS = ('tau_decaymode', 'njets')
PLOT_SUBDIRS = ('closure_in_DR', 'FF_distribution_AR', 'FF_distribution_DR', 'closure_plots')
for subdir in PLOT_SUBDIRS:
    for grouping in PLOT_GROUPINGS:
        (PLOTS_DIR / subdir / grouping).mkdir(parents=True, exist_ok=True)


def _read_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _read_labels_yaml(path):
    labels_by_channel = {}
    current_channel = None

    with open(path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            stripped = line.strip()

            if not stripped or stripped.startswith('#'):
                continue

            if stripped.endswith(':') and not line.startswith(' '):
                current_channel = stripped[:-1]
                labels_by_channel.setdefault(current_channel, {})
                continue

            if current_channel is None:
                continue

            if not line.startswith('    '):
                continue

            key_value = line.strip().split(':', 1)
            if len(key_value) != 2:
                continue

            key, value = key_value
            labels_by_channel[current_channel][key] = value.strip().strip('"').strip("'")

    return labels_by_channel


PLOTTING_CFG = _read_yaml(PLOTTING_CONFIG_PATH)
LABELS_CFG = _read_labels_yaml(LABELS_CONFIG_PATH)

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
    training_variables = load_variables(TRAINING_VAR_PATH)

    model_wjets_tdm = load_fold_combined_model(
        even_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'wjets' / 'fold_even',
        odd_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'wjets' / 'fold_odd',
    )
    model_qcd_tdm = load_fold_combined_model(
        even_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'qcd' / 'fold_even',
        odd_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'qcd' / 'fold_odd',
    )
    model_wjets_njets = load_fold_combined_model(
        even_model_path=Path(CHECKPOINT_DIR) / 'njets' / 'wjets' / 'fold_even',
        odd_model_path=Path(CHECKPOINT_DIR) / 'njets' / 'wjets' / 'fold_odd',
    )
    model_qcd_njets = load_fold_combined_model(
        even_model_path=Path(CHECKPOINT_DIR) / 'njets' / 'qcd' / 'fold_even',
        odd_model_path=Path(CHECKPOINT_DIR) / 'njets' / 'qcd' / 'fold_odd',
    )

    grouping_njets = (
        (0,),
        (1,),
        (2, 1000),
    )
    calculate_fake_factors(
        df=df,
        model_wjets=model_wjets_tdm,
        model_qcd=model_qcd_tdm,
        training_variables=training_variables,
        grouping_variable = 'njets',
        grouping_definition = grouping_njets,
        output_suffix = 'tdm',
    )

    calculate_fake_factors(
        df=df,
        model_wjets=model_wjets_njets,
        model_qcd=model_qcd_njets,
        training_variables=training_variables,
        grouping_variable = 'njets',
        grouping_definition = grouping_njets,
        output_suffix = 'njets',
    )


    calculate_fake_factor_classic(
        df = df.AR,
    )

    calculate_fake_factor_dnn(
        df = df.AR,
        grouping = 'tau_decaymode',
    )

    calculate_fake_factor_dnn(
        df = df.AR,
        grouping = 'njets',
    )

    calculate_fake_factors_in_DR_wjets(
        df,
        model_wjets_tdm,
        training_variables,
        'njets',
        grouping_njets,
        'tdm',
    )

    calculate_fake_factors_in_DR_qcd(
        df, model_qcd_tdm,
        training_variables,
        'njets',
        grouping_njets,
        'tdm',
    )

    calculate_fake_factors_in_DR_wjets(
        df,
        model_wjets_njets,
        training_variables,
        'njets',
        grouping_njets,
        'njets',
    )

    calculate_fake_factors_in_DR_qcd(
        df, model_qcd_njets,
        training_variables,
        'njets',
        grouping_njets,
        'njets',
    )


    for grouping in PLOT_GROUPINGS:
        for var in VARIABLES_SMALL:
            bins, label = get_bins_and_label(var)
            fig_w, _ = FF_closure_in_DR_wjets(
                df=df,
                var=var,
                bins=bins,
                label=label,
                grouping=grouping,
            )
            plt.savefig(PLOTS_DIR / 'closure_in_DR' / grouping / f'FF_closure_DR_wjets_{var}.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'closure_in_DR' / grouping / f'FF_closure_DR_wjets_{var}.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_w)

            fig_q, _ = FF_closure_in_DR_qcd(
                df=df,
                var=var,
                bins=bins,
                label=label,
                grouping=grouping,
            )
            plt.savefig(PLOTS_DIR / 'closure_in_DR' / grouping / f'FF_closure_DR_qcd_{var}.png', dpi=150, bbox_inches='tight')
            plt.savefig(PLOTS_DIR / 'closure_in_DR' / grouping / f'FF_closure_DR_qcd_{var}.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig_q)

        print(f'Saved closure plots for {grouping}')


    for grouping in PLOT_GROUPINGS:
        fig_ar, ax_ar = plot_fake_factors_grouped(
            df=df,
            category_title=f'split in {grouping}',
            grouping=grouping,
        )
        plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / grouping / f'plot_fake_factors_{grouping}.png', dpi=150, bbox_inches='tight')
        plt.savefig(PLOTS_DIR / 'FF_distribution_AR' / grouping / f'plot_fake_factors_{grouping}.pdf', dpi=150, bbox_inches='tight')
        plt.close(fig_ar)

        fig_dr, ax_dr = plot_fake_factors_in_dr_grouped(
            df=df,
            category_title=f'split in {grouping}',
            grouping=grouping,
        )
        plt.savefig(PLOTS_DIR / 'FF_distribution_DR' / grouping / f'plot_fake_factors_in_DR_{grouping}.png', dpi=150, bbox_inches='tight')
        plt.savefig(PLOTS_DIR / 'FF_distribution_DR' / grouping / f'plot_fake_factors_in_DR_{grouping}.pdf', dpi=150, bbox_inches='tight')
        plt.close(fig_dr)

        print(f'Saved fake-factor distributions for {grouping}')

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

    print('Saved all closure plots in njets')

# -------------------------------------------

if __name__ == '__main__':
    main()