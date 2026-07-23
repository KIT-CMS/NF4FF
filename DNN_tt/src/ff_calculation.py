from pathlib import Path
import copy
import logging
import random
import time
from typing import Literal, Union

import numpy as np
from tap import Tap
import torch as t

from classes.NeuralNetworks import load_fold_combined_model, FoldCombinedDNN
from classes.DataHandling import test_data
from classes.Loading import load_config, load_variables, load_labels, load_data
from classes.FF_calculation import calculate_fake_factors_incl, calculate_fake_factors, calculate_fake_factor_dnn, calculate_fake_factor_classic, calculate_fake_factors_in_DR_qcd


SEED = 42
logger = logging.getLogger(__name__)


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

class Args(Tap):
    taus: Literal['split', 'incl'] = 'split' # split: calc 2 FF for tau1 and tau2 | incl: calc only 1 FF
    embedding: Literal["embedding", "no_embedding"] = "embedding"
    var = "variables"
    dnn_grouped: bool = True

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

PLOT_GROUPINGS = ('tau_decaymode', 'njets')
PLOT_SUBDIRS = ('closure_in_DR', 'FF_distribution_AR', 'FF_distribution_DR', 'closure_plots')
for subdir in PLOT_SUBDIRS:
    for grouping in PLOT_GROUPINGS:
        (PLOTS_DIR / subdir / grouping).mkdir(parents=True, exist_ok=True)






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




###################
# ----- main -----#
###################

def main():

    # ----- load models from training.py -----
    if args.dnn_grouped:
        logger.info("Loading models from training.py for grouped DNN...")

        # tau decay mode
        model_tau1_tdm = load_fold_combined_model(
            even_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'tau1' / 'fold_even',
            odd_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'tau1' / 'fold_odd',
        )
        model_tau2_tdm = load_fold_combined_model(
            even_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'tau2' / 'fold_even',
            odd_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'tau2' / 'fold_odd',
        )

        # njets
        model_tau1_njets = load_fold_combined_model(
            even_model_path=Path(CHECKPOINT_DIR) / 'njets' / 'tau1' / 'fold_even',
            odd_model_path=Path(CHECKPOINT_DIR) / 'njets' / 'tau1' / 'fold_odd',
        )
        model_tau2_njets = load_fold_combined_model(
            even_model_path=Path(CHECKPOINT_DIR) / 'njets' / 'tau2' / 'fold_even',
            odd_model_path=Path(CHECKPOINT_DIR) / 'njets' / 'tau2' / 'fold_odd',
        )
    else:
        logger.info("Loading models from training.py for ungrouped DNN...")

        model_tau1 = load_fold_combined_model(
            even_model_path=Path(CHECKPOINT_DIR) / 'ungrouped' / 'tau1' / 'fold_even',
            odd_model_path=Path(CHECKPOINT_DIR) / 'ungrouped' / 'tau1' / 'fold_odd',
        )
        model_tau2 = load_fold_combined_model(
            even_model_path=Path(CHECKPOINT_DIR) / 'ungrouped' / 'tau2' / 'fold_even',
            odd_model_path=Path(CHECKPOINT_DIR) / 'ungrouped' / 'tau2' / 'fold_odd',
        )
        model_incl = load_fold_combined_model(
            even_model_path=Path(CHECKPOINT_DIR) / 'ungrouped' / 'tau_incl' / 'fold_even',
            odd_model_path=Path(CHECKPOINT_DIR) / 'ungrouped' / 'tau_incl' / 'fold_odd',
        )


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

    # ----- execution -----
    logger.info("Loading data...")
    df = load_data(DATA_PATH, MASKS_PATH)
    df_incl = load_data(DATA_PATH, MASKS_PATH_INCL)
    df_classic_jv = load_data(DATA_CLASSIC_JV_PATH, MASKS_PATH)
    df_classic_sg = load_data(DATA_CLASSIC_SG_PATH, MASKS_PATH)
    #print(df_classic.columns)
    #exit()

    training_variables = load_variables(TRAINING_VAR_PATH, args.var)

   

    # ----- calculate fake factors -----
    # classic: at the moment from jvoss smhtt ul v12
    calculate_fake_factor_classic(df_classic_jv, 'jv')
    calculate_fake_factor_classic(df_classic_sg, 'sg')


    if args.dnn_grouped:
        # tau decay mode
        logger.info("Calculating fake factors for tau decay mode...")
        calculate_fake_factors(
            df=df,
            model_tau1=model_tau1_tdm,
            model_tau2=model_tau2_tdm,
            training_variables=training_variables,
            grouping_variable = ['tau_decaymode_1', 'tau_decaymode_2'],
            grouping_definition = grouping_tdm,
            output_suffix = 'tau_dm',
        )

        # njets
        logger.info("Calculating fake factors for njets...")
        calculate_fake_factors(
            df=df,
            model_tau1=model_tau1_njets,
            model_tau2=model_tau2_njets,
            training_variables=training_variables,
            grouping_variable = 'njets',
            grouping_definition = grouping_njets,
            output_suffix = 'njets',
        )


        logger.info("Saving fake factors to df in ff_dnn_tau_dm and ff_dnn_njets columns...")
        calculate_fake_factor_dnn(
            df1 = df.AR_tau1,
            df2 = df.AR_tau2,
            grouping = 'tau_decaymode',
        )

        calculate_fake_factor_dnn(
            df1 = df.AR_tau1,
            df2 = df.AR_tau2,
            grouping = 'njets',
        )

        #the equivalent would be saving the tau1 and tau2 FF seperately
        
        # ----- calculate fake factors in DR -----
        logger.info("Calculating fake factors in DR...")
        calculate_fake_factors_in_DR_qcd(
            df=df,
            model_tau1=model_tau1_tdm,
            model_tau2=model_tau2_tdm,
            training_variables=training_variables,
            grouping_variable = ['tau_decaymode_1', 'tau_decaymode_2'],
            grouping_definition = grouping_tdm,
            output_suffix = 'tau_dm',
        )

        calculate_fake_factors_in_DR_qcd(
            df=df,
            model_tau1=model_tau1_njets,
            model_tau2=model_tau2_njets,
            training_variables=training_variables,
            grouping_variable = 'njets',
            grouping_definition = grouping_njets,
            output_suffix = 'njets',
        )

    else:
        if args.taus == 'split':
            logger.info("Calculating fake factors...")
            calculate_fake_factors(
                df=df,
                model_tau1=model_tau1,
                model_tau2=model_tau2,
                training_variables=training_variables,
            )
        elif args.taus == 'incl':
            logger.info("Calculating fake factors inclusive...")
            calculate_fake_factors_incl(
                df=df_incl,
                model=model_incl,
                training_variables=training_variables,
            )    
    

    #print(list(df.columns))
    logger.info(f"Saving main dataframe to feather file: {DATA_PATH}")
    df.to_feather(DATA_PATH)

    logger.info(f"Saving tau inclusive dataframe to feather file: {DATA_PATH}")
    df_incl.to_feather(DATA_PATH)

    logger.info(f"Saving classic dataframe with fake-factor columns to feather file: {DATA_CLASSIC_JV_PATH} and {DATA_CLASSIC_SG_PATH}")
    df_classic_jv.to_feather(DATA_CLASSIC_JV_PATH)
    df_classic_sg.to_feather(DATA_CLASSIC_SG_PATH)


if __name__ == '__main__':
    main()
    logger.info("Done.")