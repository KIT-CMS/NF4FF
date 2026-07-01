import torch as t
import numpy as np
import random
import logging
import yaml
from classes import load_variables, load_data, create_training_dataset
from classes import DNN, GroupedDNN, FoldCombinedDNN
from classes import train_dnn, save_model
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Dict, Any
import yaml
from dataclasses import is_dataclass, fields
from classes.CustomLogging import Logging
from pathlib import Path


logger = logging.getLogger(__name__)

SEED = 42
t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

DATA_PATH = '../data/data_complete.feather'
MASKS_PATH = 'configs/masks.yaml'
TRAINING_VAR_PATH = 'configs/training_variables.yaml'
NN_CONFIG_PATH = 'configs/DNN.yaml'
CHECKPOINT_DIR = 'Training_results_uncertainties'
# Dataset size control for quick studies: choose from {'full', 'half', 'quarter'}.
DATASET_SIZE = 'half'

CHECKPOINT_DIR = f'Training_results_uncertainties_{DATASET_SIZE}'

def load_variables(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    yaml_vars = config.get("variables", [])
    return yaml_vars

def load_config(path: str, cls):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return _from_dict(data, cls)

def _from_dict(data: dict, cls):
    """
    Minimal recursive dict → dataclass converter
    """

    if not is_dataclass(cls):
        return data

    kwargs = {}

    for field in fields(cls):
        value = data.get(field.name)

        if value is None:
            kwargs[field.name] = None
            continue

        # tuple conversion (important for hidden_nodes)
        if field.type == tuple or field.type == Tuple[int, ...]:
            kwargs[field.name] = tuple(value)

        # nested dataclass
        elif is_dataclass(field.type):
            kwargs[field.name] = _from_dict(value, field.type)

        else:
            kwargs[field.name] = value

    return cls(**kwargs)

@dataclass
class ModelConfig:
    hidden_nodes: Tuple[int, ...]
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


def _train_fold_model(cfg, grouping, training_var, df_sig, df_bkg, weight_column, device, checkpoint_dir, fold_label):
    train, val = create_training_dataset(
        df_sig=df_sig,
        df_bkg=df_bkg,
        training_var=training_var,
        weight_column=weight_column,
        balance=True,
        test_size=0.25,
        random_state=SEED,
    )

    base_model = DNN(
        input_nodes=train.X.shape[1],
        hidden_nodes=cfg.model.hidden_nodes,
        output_nodes=1,
        activation=cfg.model.activation,
        output_activation=cfg.model.output_activation,
        dropout=cfg.model.dropout,
        input_names=training_var,
    )

    base_model.initialize_scaler(
        shift=train.X.mean(dim=0),
        scale=train.X.std(dim=0) + 1e-6,
    )

    model = GroupedDNN(
        grouping=grouping,
        default_model=base_model,
    )

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

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    cfg = load_config(NN_CONFIG_PATH, Config)

    df = load_data(DATA_PATH, MASKS_PATH)

    # Randomly downsample once at load time without changing downstream code paths.
    size_to_frac = {
        'full': 1.0,
        'half': 0.5,
        'quarter': 0.25,
    }
    if DATASET_SIZE not in size_to_frac:
        raise ValueError(f"DATASET_SIZE must be one of {list(size_to_frac)}, got: {DATASET_SIZE}")

    sample_frac = size_to_frac[DATASET_SIZE]
    if sample_frac < 1.0:
        df._df = df._df.sample(frac=sample_frac, random_state=SEED).reset_index(drop=True)
        df._region_cache.clear()
        df._process_cache.clear()
        logger.info(
            "Applied dataset downsampling: mode=%s, frac=%.2f, n_events=%d",
            DATASET_SIZE,
            sample_frac,
            len(df._df),
        )
    else:
        logger.info("Using full dataset: n_events=%d", len(df._df))

    

    training_var = load_variables(TRAINING_VAR_PATH)

    taudm_idx = training_var.index('tau_decaymode_2')
    njets_idx = training_var.index('njets')

    grouping_taudm = {
        taudm_idx: (
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


    for seed in range(100, 201):
        # Match single-model training behavior per run, but vary RNG by seed.
        t.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        logger.info(f'Current seed: {seed}')

        for grouping, group_label in zip([grouping_taudm, grouping_njets], ['tau_decaymode', 'njets']):

            logger.info(f'Group splitting: {group_label}')

            for process in ['wjets', 'qcd', 'ttbar']:

                logger.info(f'Training process: {process}')
                
                if process == 'wjets':
                    df_sig = df.data.SR_like_wjets
                    df_bkg = df.data.AR_like_wjets
                    weight_column = 'weight_wjets'
                elif process == 'qcd':
                    df_sig = df.data.SR_like_qcd
                    df_bkg = df.data.AR_like_qcd
                    weight_column = 'weight_qcd'
                elif process == 'ttbar':
                    df_sig = df.ttbar.SR_like_ttbar
                    df_bkg = df.ttbar.AR_like_ttbar
                    weight_column = 'weight'

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
                    grouping=grouping,
                    training_var=training_var,
                    df_sig=df_sig_odd,
                    df_bkg=df_bkg_odd,
                    weight_column=weight_column,
                    device=device,
                    checkpoint_dir=CHECKPOINT_DIR,
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
                    fold_label='fold_even',
                )

                model = FoldCombinedDNN(
                    even_model=even_model,
                    odd_model=odd_model,
                    fold_id_name='event',
                )

                base_path = Path(CHECKPOINT_DIR) / group_label / process / str(seed)
                save_model(even_model, base_path / 'fold_even')
                save_model(odd_model, base_path / 'fold_odd')
                save_model(model, base_path)


if __name__ == '__main__':
    main()