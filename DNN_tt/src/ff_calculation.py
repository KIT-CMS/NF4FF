from pathlib import Path
import copy
import logging
import random
import time
from typing import Literal, Union

import numpy as np
from tap import Tap
import torch as t

from classes.NeuralNetworks import load_fold_combined_model
from classes.Loading import load_config, load_variables, load_data
from classes.FF_calculation import calculate_fake_factors_incl, calculate_fake_factors, calculate_fake_factor_classic, calculate_fake_factors_in_DR, calculate_fake_factors_in_DR_incl


SEED = 42
logger = logging.getLogger(__name__)


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

class Args(Tap):
    taus: Literal['split', 'incl'] = 'split' # split: calc 2 FF for tau1 and tau2 | incl: calc only 1 FF
    incl: Literal['and', 'or'] = 'or' # Combine tau1 and tau2 AR with and or or
    embedding: Literal["embedding", "no_embedding"] = "embedding"
    var = "variables"
    dnn_grouped: bool = False
    classic: bool = False

args = Args().parse_args()

cfg_path = load_config('/work/tapp/TauFF/NF4FF/DNN_tt/configs/config_path.yaml')

DATA_PATH = f'{cfg_path["datasets"]}/{args.embedding}/combined_data_updated.feather'
DATA_CLASSIC_JV_PATH = "/work/tapp/TauFF/NF4FF/Data/datasets/classic/combined_data_jvoss.feather"
DATA_CLASSIC_SG_PATH = "/work/tapp/TauFF/NF4FF/Data/datasets/classic/combined_data_sgiappic.feather"

MASKS_PATH = cfg_path["masks"]
MASKS_PATH_INCL = [cfg_path["masks_incl_and"], cfg_path["masks_incl_or"]]

TRAINING_VAR_PATH = cfg_path["train_var"]
CHECKPOINT_DIR = cfg_path["traininfg_results"]


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

    if args.taus=='split':
        df = load_data(DATA_PATH, MASKS_PATH)

    elif args.taus == 'incl':
        if args.incl=='and': incl = 0
        elif args.incl=='or': incl = 1
        else:
            logger.error(f'Value Error: args.incl = {args.incl}, but only accepts "and or "or".')
            exit()
        df = load_data(DATA_PATH, MASKS_PATH_INCL[incl])
    else:
        logger.error(f'Value Error: args.taus = {args.taus}, but ony allows split or incl.')
    
    #print(df.columns)
    #exit()

    training_variables = load_variables(TRAINING_VAR_PATH, args.var)

   

    # ----- calculate fake factors -----
    # classic: at the moment from jvoss smhtt ul v12

    if args.classic:
        df_classic_jv = load_data(DATA_CLASSIC_JV_PATH, MASKS_PATH)
        df_classic_sg = load_data(DATA_CLASSIC_SG_PATH, MASKS_PATH)

        #print(df_classic_jv.columns)
        #print(df_classic_sg.columns)
        #exit()
        logger.info('Calculating classic FF from jvoss...')
        calculate_fake_factor_classic(df_classic_jv, 'jv')
        logger.info('Calculating classic FF from sgiappic...')
        calculate_fake_factor_classic(df_classic_sg, 'sg')

        logger.info(f"Saving classic dataframe with fake-factor columns to feather file: {DATA_CLASSIC_JV_PATH} and {DATA_CLASSIC_SG_PATH}")
        df_classic_jv.to_feather(DATA_CLASSIC_JV_PATH)
        df_classic_sg.to_feather(DATA_CLASSIC_SG_PATH)

        return None


    if args.taus=='split' and args.dnn_grouped:
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


        #logger.info("Saving fake factors to df in ff_dnn_tau_dm and ff_dnn_njets columns...")
        #calculate_fake_factor_dnn(df1 = df.AR_tau1, df2 = df.AR_tau2, grouping = 'tau_decaymode')
        #calculate_fake_factor_dnn(df1 = df.AR_tau1, df2 = df.AR_tau2, grouping = 'njets')

        #the equivalent would be saving the tau1 and tau2 FF seperately
        
        # ----- calculate fake factors in DR -----
        logger.info("Calculating fake factors in DR...")
        calculate_fake_factors_in_DR(
            df=df,
            model_tau1=model_tau1_tdm,
            model_tau2=model_tau2_tdm,
            training_variables=training_variables,
            grouping_variable = ['tau_decaymode_1', 'tau_decaymode_2'],
            grouping_definition = grouping_tdm,
            output_suffix = 'tau_dm',
        )

        calculate_fake_factors_in_DR(
            df=df,
            model_tau1=model_tau1_njets,
            model_tau2=model_tau2_njets,
            training_variables=training_variables,
            grouping_variable = 'njets',
            grouping_definition = grouping_njets,
            output_suffix = 'njets',
        )

    elif args.taus == 'split' and not args.dnn_grouped:
        
        logger.info("Calculating fake factors...")
        calculate_fake_factors(
            df=df,
            model_tau1=model_tau1,
            model_tau2=model_tau2,
            training_variables=training_variables,
        )

        calculate_fake_factors_in_DR(
                    df=df,
                    model_tau1=model_tau1,
                    model_tau2=model_tau2,
                    training_variables=training_variables
                )

    elif args.taus == 'incl' and args.dnn_grouped:
        logger.warning('Not implemented.')

    elif args.taus == 'incl' and not args.dnn_grouped:
        logger.info("Calculating fake factors inclusive...")
        calculate_fake_factors_incl(
            df=df,
            incl = args.incl,
            model=model_incl,
            training_variables=training_variables,
        )

        # ----- calculate fake factors in DR -----
        logger.info("Calculating incusive fake factors in DR...")
        calculate_fake_factors_in_DR_incl(
            df=df,
            incl=args.incl,
            model=model_incl,
            training_variables=training_variables
        )
    

    #print(list(df.columns))

    if args.taus == 'split':
        logger.info(f"Saving main dataframe to feather file: {DATA_PATH}")
        df.to_feather(DATA_PATH)
    elif args.taus == 'incl':
        logger.info(f"Saving tau inclusive dataframe to feather file: {DATA_PATH}")
        df.to_feather(DATA_PATH)
    else:
        logger.warning(f'df could not be saved. taus = {args.taus}, but accepts only "split" and "incl"')


if __name__ == '__main__':
    main()
    logger.info("Done.")