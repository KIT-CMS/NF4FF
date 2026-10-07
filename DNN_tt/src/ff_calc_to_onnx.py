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
    embedding: Literal["embedding", "no_embedding"] = "embedding"
    var = "variables"

    dnn_grouped: bool = True
args = Args().parse_args()

cfg_path = load_config('/work/tapp/TauFF/NF4FF/DNN_tt/configs/config_path.yaml')

DATA_PATH = f'{cfg_path["datasets"]}/{args.embedding}/combined_data_updated.feather'
MASKS_PATH = cfg_path["masks"]
MASKS_PATH_INCL = [cfg_path["masks_incl_and"], cfg_path["masks_incl_or"], cfg_path["masks_incl_andor"]]
MASKS_PATH_3SPLIT = cfg_path["masks_3split"]
TRAINING_VAR_PATH = cfg_path["train_var"]
NN_CONFIG_PATH = cfg_path["DNN"]
CHECKPOINT_DIR = cfg_path["training_results"]


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


def main():
    logger.info("Starting FF calculation conversion to ONNX...")

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    cfg = load_config(NN_CONFIG_PATH, Config)