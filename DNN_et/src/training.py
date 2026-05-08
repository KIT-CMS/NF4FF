import torch as t
import numpy as np
import random
import yaml
from classes import load_data, create_training_dataset
from classes import DNN, GroupedDNN
from classes import train_dnn, save_model
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Dict, Any
import yaml
from dataclasses import is_dataclass, fields
from classes.CustomLogging import Logging
SEED = 42


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

DATA_PATH = '../data/data_complete.feather'
MASKS_PATH = 'configs/masks.yaml'
TRAINING_VAR_PATH = 'configs/training_variables.yaml'
NN_CONFIG_PATH = 'configs/DNN.yaml'
CHECKPOINT_DIR = 'Training_Results'



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



def main():

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    cfg = load_config(NN_CONFIG_PATH, Config)

    df = load_data(DATA_PATH, MASKS_PATH)
    
    training_var = load_variables(TRAINING_VAR_PATH)
    
    train, val = create_training_dataset(
        df_sig = df.data.SR_like_wjets,
        df_bkg = df.data.AR_like_wjets,
        training_var = training_var,
        weight_column = 'weight_wjets',
        balance = True,
        test_size = 0.25,
        random_state = SEED
    )

    taudm_idx = training_var.index('tau_decaymode_1')

    grouping = {
        taudm_idx: (
            (0,),
            (1,),
            (10,),
            (11,),
        )
    }

    base_model = DNN(
        input_nodes = train.X.shape[1],
        hidden_nodes = cfg.model.hidden_nodes,
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
        shift = train.X.mean(dim = 0),
        scale = train.X.std(dim = 0) + 1e-6,
    )
    model, best_loss = train_dnn(
        model = model,
        train = train,
        val = val,
        epochs = cfg.training.epochs,
        lr = cfg.training.lr,
        loss_fn=None,
        device = device,
        checkpoint_dir=CHECKPOINT_DIR,
        scheduler_patience=cfg.scheduler.patience,
        early_stopping_patience=cfg.scheduler.early_stopping_patience,
        scheduler_factor=cfg.scheduler.factor,
        min_delta=cfg.scheduler.min_delta,
        min_lr=cfg.scheduler.min_lr,
    )

    save_model(model, "src/Training_results/best_model")





if __name__ == '__main__':
    main()