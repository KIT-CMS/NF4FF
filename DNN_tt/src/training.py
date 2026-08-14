from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Union, Tuple, Literal

import numpy as np
import random
from tap import Tap
import torch as t

from classes.DataHandling import create_training_dataset
from classes.NeuralNetworks import DNN, GroupedDNN, FoldCombinedDNN, save_model
from classes.Training import train_dnn
from classes.Loading import load_config, load_variables, load_data



SEED = 42
logger = logging.getLogger(__name__)


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

class Args(Tap):
    taus: Literal['split', 'incl'] = 'incl' # split: calc 2 FF for tau1 and tau2 | incl: calc only 1 FF
    incl: Literal['and', 'or'] = 'or' # Combine tau1 and tau2 AR with and or or
    embedding: Literal["embedding", "no_embedding"] = "embedding"
    var = "variables"
    dnn_grouped: bool = True

args = Args().parse_args()

cfg_path = load_config('/work/tapp/TauFF/NF4FF/DNN_tt/configs/config_path.yaml')

DATA_PATH = f'{cfg_path["datasets"]}/{args.embedding}/combined_data_updated.feather'
MASKS_PATH = cfg_path["masks"]
MASKS_PATH_INCL = [cfg_path["masks_incl_and"], cfg_path["masks_incl_or"]]
TRAINING_VAR_PATH = cfg_path["train_var"]
NN_CONFIG_PATH = cfg_path["DNN"]
CHECKPOINT_DIR = cfg_path["traininfg_results"]

@dataclass
class ModelConfig:
    hidden_nodes: Tuple[int, ...]
    hidden_nodes_incl: Tuple[int, ...]
    output_nodes: int

    activation: str = "ReLU"
    output_activation: str = "Sigmoid"

    dropout: Union[float, Tuple[float, ...]] = 0.1

@dataclass
class TrainingConfig:
    epochs: int = 50
    lr: float = 1e-3
    loss: str = "BCE"

@dataclass
class SchedulerConfig:
    patience: int = 10
    early_stopping_patience: int = 20
    factor: float = 0.1
    min_delta: float = 1.0e-4
    min_lr: float = 1e-6

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    scheduler: SchedulerConfig


def _train_fold_model(cfg, grouping, training_var, df_sig, df_bkg, weight_column, device, checkpoint_dir, taus, fold_label):
    train, val = create_training_dataset(
        df_sig=df_sig,
        df_bkg=df_bkg,
        training_var=training_var,
        weight_column=weight_column,
        balance=True,
        test_size=0.25,
        random_state=SEED,
    )

    if taus == 'incl':
        base_model = DNN(
            input_nodes=train.X.shape[1],
            hidden_nodes=cfg.model.hidden_nodes_incl,
            output_nodes=1,
            activation=cfg.model.activation,
            output_activation=cfg.model.output_activation,
            dropout=cfg.model.dropout,
            input_names=training_var,
        )
    elif taus == 'split':
        base_model = DNN(
            input_nodes=train.X.shape[1],
            hidden_nodes=cfg.model.hidden_nodes,
            output_nodes=1,
            activation=cfg.model.activation,
            output_activation=cfg.model.output_activation,
            dropout=cfg.model.dropout,
            input_names=training_var,
        )
    else:
        logger.error(f'Value Error: args.taus = {taus}, but ony allows split or incl.')
        exit()

    base_model.initialize_scaler(
        shift=train.X.mean(dim=0),
        scale=train.X.std(dim=0) + 1e-6,
    )


    if args.dnn_grouped:    
        model = GroupedDNN(
            grouping=grouping,
            default_model=base_model,
        )
    else:
        model = base_model

    model, best_loss = train_dnn(
        model=model,
        train=train,
        val=val,
        epochs=cfg.training.epochs,
        lr=cfg.training.lr,
        loss_fn=None,
        device=device,
        checkpoint_dir=checkpoint_dir,
        scheduler_patience=cfg.scheduler.patience,
        early_stopping_patience=cfg.scheduler.early_stopping_patience,
        scheduler_factor=cfg.scheduler.factor,
        min_delta=cfg.scheduler.min_delta,
        min_lr=cfg.scheduler.min_lr,
    )

    return model


def main():
    print('start')

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    cfg = load_config(NN_CONFIG_PATH, Config)
    if args.taus == 'split':
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
        exit()

    training_var = load_variables(TRAINING_VAR_PATH, args.var)

    taudm1_idx = training_var.index('tau_decaymode_1')
    taudm2_idx = training_var.index('tau_decaymode_2')
    njets_idx = training_var.index('njets')


    grouping_taudm1 = {
        taudm1_idx: (
            (0,),
            (1,),
            (10,),
            (11,),
        )
    }

    grouping_taudm2 = {
        taudm2_idx: (
            (0,),
            (1,),
            (10,),
            (11,),
        )
    }

    grouping_njets = {
        njets_idx: (
            (0,),
            (1,),
            (2, 1000),
        )
    }


    if args.taus=='split' and args.dnn_grouped:
        logger.info('Training uses the grouped DNN.')
        for grouping, group_label in zip([[grouping_taudm1, grouping_taudm2], [grouping_njets]], ['tau_decaymode', 'njets']):
            logger.info(f'Group splitting: {group_label}')
            
            i = 0
            for process in ['tau1', 'tau2']:

                logger.info(f'Training process: {process}')
                
                if process == 'tau1':
                    df_sig = df.data.SR_like
                    df_bkg = df.data.AR_like_tau1
                    weight_column = 'weight_qcd'

                elif process == 'tau2':
                    df_sig = df.data.SR_like
                    df_bkg = df.data.AR_like_tau2
                    weight_column = 'weight_qcd'

                df_sig_plain = df_sig.events
                df_bkg_plain = df_bkg.events
                df_sig_even = df_sig_plain[df_sig_plain['event']%2 == 0]
                df_sig_odd  = df_sig_plain[df_sig_plain['event']%2 == 1]
                df_bkg_even = df_bkg_plain[df_bkg_plain['event']%2 == 0]
                df_bkg_odd  = df_bkg_plain[df_bkg_plain['event']%2 == 1]

                logger.info(
                    "%s/%s fold sizes: even=%d (sig=%d, bkg=%d), odd=%d (sig=%d, bkg=%d)",
                    group_label,
                    process,
                    len(df_sig_even) + len(df_bkg_even),
                    len(df_sig_even),
                    len(df_bkg_even),
                    len(df_sig_odd) + len(df_bkg_odd),
                    len(df_sig_odd),
                    len(df_bkg_odd),
                )

                # even_model: trained on odd events, applied to even events
                even_model = _train_fold_model(
                    cfg=cfg,
                    grouping=grouping[i],
                    training_var=training_var,
                    df_sig=df_sig_odd,
                    df_bkg=df_bkg_odd,
                    weight_column=weight_column,
                    device=device,
                    checkpoint_dir=CHECKPOINT_DIR,
                    taus=args.taus,
                    fold_label='fold_odd',
                )

                # odd_model: trained on even events, applied to odd events
                odd_model = _train_fold_model(
                    cfg=cfg,
                    grouping=grouping[i],
                    training_var=training_var,
                    df_sig=df_sig_even,
                    df_bkg=df_bkg_even,
                    weight_column=weight_column,
                    device=device,
                    checkpoint_dir=CHECKPOINT_DIR,
                    taus=args.taus,
                    fold_label='fold_even',
                )

                model = FoldCombinedDNN(
                    even_model=even_model,
                    odd_model=odd_model,
                    fold_id_name='event',
                )

                base_path = Path(CHECKPOINT_DIR) / group_label / process
                save_model(even_model, base_path / 'fold_even')
                save_model(odd_model, base_path / 'fold_odd')
                save_model(model, base_path)

                if group_label == 'tau_decaymode':
                    i += 1

    elif args.taus=='split' and not args.dnn_grouped:
        logger.info('Training uses the ungrouped DNN.')
        for process in ['tau1', 'tau2']:
            logger.info(f'Training process: {process}')
            
            if process == 'tau1':
                df_sig = df.data.SR_like
                df_bkg = df.data.AR_like_tau1
                weight_column = 'weight_qcd'

            elif process == 'tau2':
                df_sig = df.data.SR_like
                df_bkg = df.data.AR_like_tau2
                weight_column = 'weight_qcd'

            df_sig_plain = df_sig.events
            df_bkg_plain = df_bkg.events
            df_sig_even = df_sig_plain[df_sig_plain['event']%2 == 0]
            df_sig_odd  = df_sig_plain[df_sig_plain['event']%2 == 1]
            df_bkg_even = df_bkg_plain[df_bkg_plain['event']%2 == 0]
            df_bkg_odd  = df_bkg_plain[df_bkg_plain['event']%2 == 1]

            logger.info(
                "%s fold sizes: even=%d (sig=%d, bkg=%d), odd=%d (sig=%d, bkg=%d)",
                process,
                len(df_sig_even) + len(df_bkg_even),
                len(df_sig_even),
                len(df_bkg_even),
                len(df_sig_odd) + len(df_bkg_odd),
                len(df_sig_odd),
                len(df_bkg_odd),
            )

            # even_model: trained on odd events, applied to even events
            even_model = _train_fold_model(
                cfg=cfg,
                grouping=None,
                training_var=training_var,
                df_sig=df_sig_odd,
                df_bkg=df_bkg_odd,
                weight_column=weight_column,
                device=device,
                checkpoint_dir=CHECKPOINT_DIR,
                taus=args.taus,
                fold_label='fold_odd',
            )

            # odd_model: trained on even events, applied to odd events
            odd_model = _train_fold_model(
                cfg=cfg,
                grouping=None,
                training_var=training_var,
                df_sig=df_sig_even,
                df_bkg=df_bkg_even,
                weight_column=weight_column,
                device=device,
                checkpoint_dir=CHECKPOINT_DIR,
                taus=args.taus,
                fold_label='fold_even',
            )

            model = FoldCombinedDNN(
                even_model=even_model,
                odd_model=odd_model,
                fold_id_name='event',
            )

            base_path = Path(CHECKPOINT_DIR) / 'ungrouped' / process
            save_model(even_model, base_path / 'fold_even')
            save_model(odd_model, base_path / 'fold_odd')
            save_model(model, base_path)

    elif args.taus == 'incl' and args.dnn_grouped:
        logger.info('Training uses the grouped DNN tau inclusive.')

        for grouping, group_label in zip([grouping_njets], ['njets']):
            logger.info(f'Group splitting: {group_label}')

            logger.info(f'Training process: tau inclusive')
            
            df_sig = df.data.SR_like
            df_bkg = df.data.AR_like
            weight_column = 'weight_qcd'
    
            df_sig_plain = df_sig.events
            df_bkg_plain = df_bkg.events
            df_sig_even = df_sig_plain[df_sig_plain['event']%2 == 0]
            df_sig_odd  = df_sig_plain[df_sig_plain['event']%2 == 1]
            df_bkg_even = df_bkg_plain[df_bkg_plain['event']%2 == 0]
            df_bkg_odd  = df_bkg_plain[df_bkg_plain['event']%2 == 1]

            logger.info(
                "%s fold sizes: even=%d (sig=%d, bkg=%d), odd=%d (sig=%d, bkg=%d)",
                group_label,
                len(df_sig_even) + len(df_bkg_even),
                len(df_sig_even),
                len(df_bkg_even),
                len(df_sig_odd) + len(df_bkg_odd),
                len(df_sig_odd),
                len(df_bkg_odd),
            )

            # even_model: trained on odd events, applied to even events
            even_model = _train_fold_model(
                cfg=cfg,
                grouping=grouping,
                training_var=training_var,
                df_sig=df_sig_odd,
                df_bkg=df_bkg_odd,
                weight_column=weight_column,
                device=device,
                checkpoint_dir=CHECKPOINT_DIR,
                taus=args.taus,
                fold_label='fold_odd',
            )

            # odd_model: trained on even events, applied to odd events
            odd_model = _train_fold_model(
                cfg=cfg,
                grouping=grouping,
                training_var=training_var,
                df_sig=df_sig_even,
                df_bkg=df_bkg_even,
                weight_column=weight_column,
                device=device,
                checkpoint_dir=CHECKPOINT_DIR,
                taus=args.taus,
                fold_label='fold_even',
            )

            model = FoldCombinedDNN(
                even_model=even_model,
                odd_model=odd_model,
                fold_id_name='event',
            )

            base_path = Path(CHECKPOINT_DIR) / group_label / 'tau_incl'
            save_model(even_model, base_path / 'fold_even')
            save_model(odd_model, base_path / 'fold_odd')
            save_model(model, base_path)

    elif args.taus == 'incl' and not args.dnn_grouped:
        logger.info('Training uses the ungrouped DNN tau inclusive.')
            
        
        df_sig = df.data.SR_like
        df_bkg = df.data.AR_like
        weight_column = 'weight_qcd'

        df_sig_plain = df_sig.events
        df_bkg_plain = df_bkg.events
        df_sig_even = df_sig_plain[df_sig_plain['event']%2 == 0]
        df_sig_odd  = df_sig_plain[df_sig_plain['event']%2 == 1]
        df_bkg_even = df_bkg_plain[df_bkg_plain['event']%2 == 0]
        df_bkg_odd  = df_bkg_plain[df_bkg_plain['event']%2 == 1]

        logger.info(
            "Tau inclusive fold sizes: even=%d (sig=%d, bkg=%d), odd=%d (sig=%d, bkg=%d)",
            len(df_sig_even) + len(df_bkg_even),
            len(df_sig_even),
            len(df_bkg_even),
            len(df_sig_odd) + len(df_bkg_odd),
            len(df_sig_odd),
            len(df_bkg_odd),
        )

        # even_model: trained on odd events, applied to even events
        even_model = _train_fold_model(
            cfg=cfg,
            grouping=None,
            training_var=training_var,
            df_sig=df_sig_odd,
            df_bkg=df_bkg_odd,
            weight_column=weight_column,
            device=device,
            checkpoint_dir=CHECKPOINT_DIR,
            taus=args.taus,
            fold_label='fold_odd',
        )

        # odd_model: trained on even events, applied to odd events
        odd_model = _train_fold_model(
            cfg=cfg,
            grouping=None,
            training_var=training_var,
            df_sig=df_sig_even,
            df_bkg=df_bkg_even,
            weight_column=weight_column,
            device=device,
            checkpoint_dir=CHECKPOINT_DIR,
            taus=args.taus,
            fold_label='fold_even',
        )

        model = FoldCombinedDNN(
            even_model=even_model,
            odd_model=odd_model,
            fold_id_name='event',
        )

        base_path = Path(CHECKPOINT_DIR) / 'ungrouped' / 'tau_incl'
        save_model(even_model, base_path / 'fold_even')
        save_model(odd_model, base_path / 'fold_odd')
        save_model(model, base_path)
    
    else:
        logger.error(f'Value Error: args.taus = {args.taus}, but ony allows split or incl.')

if __name__ == '__main__':
    main()