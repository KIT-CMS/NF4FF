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

SEED = 42
logger = logging.getLogger(__name__)


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

DATA_PATH = '../data/data_complete.feather'
MASKS_PATH = 'configs/masks.yaml'
TRAINING_VAR_PATH = 'configs/training_variables.yaml'
NN_CONFIG_PATH = 'configs/DNN.yaml'
CHECKPOINT_DIR = 'Training_results'



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

    model = GroupedDNN(
        grouping=grouping,
        default_model=base_model,
    )

    model.initialize_scaler(
        shift=train.X.mean(dim=0),
        scale=train.X.std(dim=0) + 1e-6,
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

    for grouping, group_label in zip([grouping_taudm, grouping_njets], ['tau_decaymode', 'njets']):

        print(f'Group splitting: {group_label}')

        for process in ['wjets', 'qcd']:

            if process == 'wjets':
                df_sig = df.data.SR_like_wjets
                df_bkg = df.data.AR_like_wjets
                weight_column = 'weight_wjets'
            elif process == 'qcd':
                df_sig = df.data.SR_like_qcd
                df_bkg = df.data.AR_like_qcd
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

            base_path = Path(CHECKPOINT_DIR) / group_label / process
            save_model(even_model, base_path / 'fold_even')
            save_model(odd_model, base_path / 'fold_odd')
            save_model(model, base_path)


if __name__ == '__main__':
    main()