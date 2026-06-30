from pathlib import Path

from classes import load_variables, load_data, load_model, load_fold_combined_model, test_data
from classes import calculate_fake_factors, calculate_fake_factor_dnn, calculate_fake_factor_classic
from classes import calculate_fake_factors_in_DR_wjets, calculate_fake_factors_in_DR_qcd, calculate_fake_factors_in_DR_ttbar
from classes import (
    FF_closure_in_DR_wjets, 
    FF_closure_in_DR_qcd, 
    FF_closure_in_DR_ttbar,
    FF_closure_in_DR_ttbar_MC,
    plot_fake_factors_in_DR, 
    plot_fake_factors,
    write_features,
	append_features,
    update_features,
    )
from classes import(
    calculate_fake_factor_mean_std,
    calculate_fake_factor_mean_std_dropout_mask_variation,
    calculate_fake_factor_mean_std_in_DR,
    calculate_fake_factor_mean_std_in_DR_dropout_mask_variation,
)
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
from classes import FoldCombinedDNN, load_fold_combined_model
import time
import torch as t
from pathlib import Path

from classes import DNN


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



def load_models(checkpoint_dir,
                seeds,
                process = 'wjets',
                ):
    models = []

    for seed in seeds:
        model = load_fold_combined_model(
            even_model_path=(
                Path(checkpoint_dir)
                / 'tau_decaymode'
                / process
                / str(seed)
                / 'fold_even'
            ),
            odd_model_path=(
                Path(checkpoint_dir)
                / 'tau_decaymode'
                / process
                / str(seed)
                / 'fold_odd'
            ),
        )

        model.eval()
        models.append(model)

    return models

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
            indent = len(line) - len(line.lstrip(' '))

            if not stripped or stripped.startswith('#'):
                continue

            # Be tolerant if the channel key is accidentally indented by one space.
            if stripped.endswith(':') and ':' not in stripped[:-1] and indent <= 1:
                current_channel = stripped[:-1]
                labels_by_channel.setdefault(current_channel, {})
                continue

            if current_channel is None:
                continue

            if indent < 4:
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



model_wjets_tdm = load_fold_combined_model(
    even_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'wjets' / 'fold_even',
    odd_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'wjets' / 'fold_odd',
)
model_qcd_tdm = load_fold_combined_model(
    even_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'qcd' / 'fold_even',
    odd_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'qcd' / 'fold_odd',
)

model_ttbar_tdm = load_fold_combined_model(
    even_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'ttbar' / 'fold_even',
    odd_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'ttbar' / 'fold_odd',
)


models_wjets_full = load_models(
    checkpoint_dir='../Training_results_uncertainties_full_data',
    seeds = range(100, 200),
    process = 'wjets',
)

models_qcd_full = load_models(
    checkpoint_dir='../Training_results_uncertainties_full_data',
    seeds = range(100, 200),
    process = 'qcd',
)

models_ttbar_full = load_models(
    checkpoint_dir='../Training_results_uncertainties_full_data',
    seeds = range(100, 200),
    process = 'ttbar',
)

models_wjets_half = load_models(
    checkpoint_dir='../Training_results_uncertainties_half_data',
    seeds = range(100, 200),
    process = 'wjets',
)

models_qcd_half = load_models(
    checkpoint_dir='../Training_results_uncertainties_half_data',
    seeds = range(100, 200),
    process = 'qcd',
)

models_ttbar_half = load_models(
    checkpoint_dir='../Training_results_uncertainties_half_data',
    seeds = range(100, 200),
    process = 'ttbar',
)

models_wjets_quarter = load_models(
    checkpoint_dir='../Training_results_uncertainties_quarter_data',
    seeds = range(100, 200),
    process = 'wjets',
)

models_qcd_quarter = load_models(
    checkpoint_dir='../Training_results_uncertainties_quarter_data',
    seeds = range(100, 200),
    process = 'qcd',
)

models_ttbar_quarter = load_models(
    checkpoint_dir='../Training_results_uncertainties_quarter_data',
    seeds = range(100, 200),
    process = 'ttbar',
)



# ---------------- execution part

df = load_data(DATA_PATH, MASKS_PATH)
training_variables = load_variables(TRAINING_VAR_PATH)

grouping_njets = (
    (0,),
    (1,),
    (2, 1000),
)


##### determination of fake factor mean and std via dropout mask variation in the different DRs

calculate_fake_factor_mean_std_in_DR_dropout_mask_variation(
    df = df,
    model = model_wjets_tdm,
    training_variables = training_variables,
    grouping_variable = 'njets',
    grouping_definition = grouping_njets,
    process='wjets',
    output_mean='ff_wjets_mean_dmv',
    output_std='ff_wjets_std_dmv',
)

calculate_fake_factor_mean_std_in_DR_dropout_mask_variation(
    df = df,
    model = model_qcd_tdm,
    training_variables = training_variables,
    grouping_variable = 'njets',
    grouping_definition = grouping_njets,
    process='qcd',
    output_mean='ff_qcd_mean_dmv',
    output_std='ff_qcd_std_dmv',
)

calculate_fake_factor_mean_std_in_DR_dropout_mask_variation(
    df = df,
    model = model_ttbar_tdm,
    training_variables = training_variables,
    grouping_variable = 'njets',
    grouping_definition = grouping_njets,
    process='ttbar',
    output_mean='ff_ttbar_mean_dmv',
    output_std='ff_ttbar_std_dmv',
)


# for full dataset
# determination of ff mean and std for DRs

calculate_fake_factor_mean_std_in_DR(
    df= df,
    models= models_wjets_full,
    training_variables=training_variables,
    grouping_variable='njets',
    grouping_definition=grouping_njets,
    process = 'wjets',
    output_mean='ff_wjets_mean',
    output_std='ff_wjets_std',
)


calculate_fake_factor_mean_std_in_DR(
    df= df,
    models= models_qcd_full,
    training_variables=training_variables,
    grouping_variable='njets',
    grouping_definition=grouping_njets,
    process = 'qcd',
    output_mean='ff_qcd_mean',
    output_std='ff_qcd_std',
)

calculate_fake_factor_mean_std_in_DR(
    df= df,
    models= models_ttbar_full,
    training_variables=training_variables,
    grouping_variable='njets',
    grouping_definition=grouping_njets,
    process = 'ttbar',
    output_mean='ff_ttbar_mean',
    output_std='ff_ttbar_std',
)


# for half dataset
# determination of ff mean and std for DRs

calculate_fake_factor_mean_std_in_DR(
    df= df,
    models= models_wjets_half,
    training_variables=training_variables,
    grouping_variable='njets',
    grouping_definition=grouping_njets,
    process = 'wjets',
    output_mean='ff_wjets_mean_half',
    output_std='ff_wjets_std_half',
)

'''
calculate_fake_factor_mean_std_in_DR(
    df= df,
    models= models_qcd_half,
    training_variables=training_variables,
    grouping_variable='njets',
    grouping_definition=grouping_njets,
    process = 'qcd',
    output_mean='ff_qcd_mean_half',
    output_std='ff_qcd_std_half',
)

calculate_fake_factor_mean_std_in_DR(
    df= df,
    models= models_ttbar_half,
    training_variables=training_variables,
    grouping_variable='njets',
    grouping_definition=grouping_njets,
    process = 'ttbar',
    output_mean='ff_ttbar_mean_half',
    output_std='ff_ttbar_std_half',
)
'''
# for quarter dataset
# determination of ff mean and std for DRs

calculate_fake_factor_mean_std_in_DR(
    df= df,
    models= models_wjets_quarter,
    training_variables=training_variables,
    grouping_variable='njets',
    grouping_definition=grouping_njets,
    process = 'wjets',
    output_mean='ff_wjets_mean_quarter',
    output_std='ff_wjets_std_quarter',
)

'''
calculate_fake_factor_mean_std_in_DR(
    df= df,
    models= models_qcd_quarter,
    training_variables=training_variables,
    grouping_variable='njets',
    grouping_definition=grouping_njets,
    process = 'qcd',
    output_mean='ff_qcd_mean_quarter',
    output_std='ff_qcd_std_quarter',
)

calculate_fake_factor_mean_std_in_DR(
    df= df,
    models= models_ttbar_quarter,
    training_variables=training_variables,
    grouping_variable='njets',
    grouping_definition=grouping_njets,
    process = 'ttbar',
    output_mean='ff_ttbar_mean_quarter',
    output_std='ff_ttbar_std_quarter',
)

'''

write_features(
    df.events,
    "/work/mmoser/NF4FF/data/features/fake_factor_unc.feather",
    {
        "ff_wjets_mean": df.AR_like_wjets.ff_wjets_mean,
        "ff_wjets_std": df.AR_like_wjets.ff_wjets_std,
        "ff_qcd_mean": df.AR_like_qcd.ff_qcd_mean,
        "ff_qcd_std": df.AR_like_qcd.ff_qcd_std,
        "ff_ttbar_mean": df.AR_like_ttbar.ff_ttbar_mean,
        "ff_ttbar_std": df.AR_like_ttbar.ff_ttbar_std,
        "ff_wjets_mean_dmv": df.AR_like_wjets.ff_wjets_mean_dmv,
        "ff_wjets_std_dmv": df.AR_like_wjets.ff_wjets_std_dmv,
        "ff_qcd_mean_dmv": df.AR_like_qcd.ff_qcd_mean_dmv,
        "ff_qcd_std_dmv": df.AR_like_qcd.ff_qcd_std_dmv,
        "ff_ttbar_mean_dmv": df.AR_like_ttbar.ff_ttbar_mean_dmv,
        "ff_ttbar_std_dmv": df.AR_like_ttbar.ff_ttbar_std_dmv,
	}
)

write_features(
    df.events,
    "/work/mmoser/NF4FF/data/features/fake_factor_unc_diff_train_size.feather",
    {
        "ff_wjets_mean_full": df.AR_like_wjets.ff_wjets_mean,
        "ff_wjets_std_full": df.AR_like_wjets.ff_wjets_std,
        "ff_wjets_mean_half": df.AR_like_wjets.ff_wjets_mean_half,
        "ff_wjets_std_half": df.AR_like_wjets.ff_wjets_std_half,
        "ff_wjets_mean_quarter": df.AR_like_wjets.ff_wjets_mean_quarter,
        "ff_wjets_std_quarter": df.AR_like_wjets.ff_wjets_std_quarter,
        
	}
)