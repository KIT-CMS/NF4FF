import csv
import json
import logging
import random
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as t
import yaml

from classes import (
    DNN,
    FoldCombinedDNN,
    create_training_dataset,
    load_data,
    save_model,
    train_dnn_squeezed_loss,
)
from groupings import GROUPING_NAMES
from training_squeezed_loss import squeezing_label, squeezing_limit_from_probability


SEED = 42
logger = logging.getLogger(__name__)

t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataframe_complete.feather"
MASKS_PATH = PROJECT_ROOT / "configs" / "masks.yaml"
TRAINING_VAR_PATH = PROJECT_ROOT / "configs" / "training_variables.yaml"
NN_CONFIG_PATH = PROJECT_ROOT / "configs" / "DNN.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "Training_results_squeezed_single_dnn"
REDUCED_WEIGHT_DIR = (
    PROJECT_ROOT / "data" / "features" / "reduced_dataset"
)
PROCESSES = ("wjets", "qcd", "ttbar")


def load_variables(yaml_path):
    with open(yaml_path, "r") as stream:
        return (yaml.safe_load(stream) or {}).get("variables", [])


def _from_dict(data, cls):
    if not is_dataclass(cls):
        return data
    kwargs = {}
    for field in fields(cls):
        value = data.get(field.name)
        if value is None:
            kwargs[field.name] = None
        elif field.type == tuple or field.type == Tuple[int, ...]:
            kwargs[field.name] = tuple(value)
        elif is_dataclass(field.type):
            kwargs[field.name] = _from_dict(value, field.type)
        else:
            kwargs[field.name] = value
    return cls(**kwargs)


def load_config(path, cls):
    with open(path, "r") as stream:
        return _from_dict(yaml.safe_load(stream) or {}, cls)


def save_loss_history(history: List[Dict[str, Any]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "loss_history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    if history:
        with open(output_dir / "loss_history.csv", "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


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


def _prepare_training_frame(
    frame,
    training_variables,
    source_weight,
    weight_column,
    label,
):
    frame = frame.copy()
    if source_weight is not None:
        frame[weight_column] = frame[source_weight]

    required_columns = [*training_variables, weight_column]
    finite = np.isfinite(
        frame[required_columns].to_numpy(dtype=np.float64)
    ).all(axis=1)
    if not finite.all():
        logger.warning(
            "%s: dropping %d/%d non-finite rows.",
            label,
            int((~finite).sum()),
            len(frame),
        )
    frame = frame.loc[finite].copy()
    if frame.empty:
        raise ValueError(f"{label}: no finite training rows remain.")
    return frame


def _train_fold_model(
    *,
    cfg,
    training_variables,
    signal,
    background,
    weight_column,
    balance_with_absolute_yields,
    squeezing_limit,
    device,
    checkpoint_dir,
    fold_label,
):
    train, validation = create_training_dataset(
        df_sig=signal,
        df_bkg=background,
        training_var=training_variables,
        weight_column=weight_column,
        balance=True,
        balance_column=None,
        balance_with_absolute_yields=balance_with_absolute_yields,
        test_size=0.25,
        random_state=SEED,
    )

    model = DNN(
        input_nodes=train.X.shape[1],
        hidden_nodes=cfg.model.hidden_nodes,
        output_nodes=1,
        activation=cfg.model.activation,
        output_activation=cfg.model.output_activation,
        dropout=cfg.model.dropout,
        input_names=training_variables,
    )
    model.initialize_scaler(
        shift=train.X.mean(dim=0),
        scale=train.X.std(dim=0) + 1e-6,
    )
    model, _, history = train_dnn_squeezed_loss(
        model=model,
        train=train,
        val=validation,
        squeezing_limit=squeezing_limit,
        epochs=cfg.training.epochs,
        lr=cfg.training.lr,
        device=device,
        checkpoint_dir=checkpoint_dir,
        scheduler_patience=cfg.scheduler.patience,
        early_stopping_patience=cfg.scheduler.early_stopping_patience,
        scheduler_factor=cfg.scheduler.factor,
        min_delta=cfg.scheduler.min_delta,
        min_lr=cfg.scheduler.min_lr,
    )
    for row in history:
        row["fold_label"] = fold_label
    return model, history


def train_squeezed_single_dnn_models(
    squeezing: Optional[float] = None,
    reduced_weight_grouping: str = "tau_decaymode_2_alt",
    data_path: Union[str, Path] = DATA_PATH,
    masks_path: Union[str, Path] = MASKS_PATH,
    training_var_path: Union[str, Path] = TRAINING_VAR_PATH,
    nn_config_path: Union[str, Path] = NN_CONFIG_PATH,
    checkpoint_dir: Union[str, Path] = CHECKPOINT_DIR,
    reduced_weight_dir: Union[str, Path] = REDUCED_WEIGHT_DIR,
):
    if reduced_weight_grouping not in GROUPING_NAMES:
        raise ValueError(
            "Unknown reduced-weight grouping: "
            f"{reduced_weight_grouping}"
        )

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    cfg = load_config(nn_config_path, Config)
    df = load_data(data_path, masks_path)
    training_variables = load_variables(training_var_path)
    checkpoint_dir = Path(checkpoint_dir)
    reduced_weight_dir = Path(reduced_weight_dir)
    output_dir = (
        checkpoint_dir
        / reduced_weight_grouping
        / squeezing_label(squeezing)
    )
    squeezing_limit = squeezing_limit_from_probability(squeezing)

    for process in PROCESSES:
        if process == "ttbar":
            source_weight = None
            weight_column = "weight"
            signal = df.ttbar.SR_like_ttbar.events
            background = df.ttbar.AR_like_ttbar.events
        else:
            source_weight = (
                f"reduced_weight_{process}_"
                f"{reduced_weight_grouping}_nominal"
            )
            weight_column = f"weight_{process}"
            df.load_feature_file(
                reduced_weight_dir
                / process
                / f"reduced_weight_{reduced_weight_grouping}.feather"
            )
            signal = getattr(df.data, f"SR_like_{process}").events
            background = getattr(df.data, f"AR_like_{process}").events

        signal = _prepare_training_frame(
            signal,
            training_variables,
            source_weight,
            weight_column,
            f"single_dnn/{process}/SR-like",
        )
        background = _prepare_training_frame(
            background,
            training_variables,
            source_weight,
            weight_column,
            f"single_dnn/{process}/AR-like",
        )

        fold_frames = {
            "fold_even": (
                signal[signal["event"] % 2 == 1],
                background[background["event"] % 2 == 1],
            ),
            "fold_odd": (
                signal[signal["event"] % 2 == 0],
                background[background["event"] % 2 == 0],
            ),
        }
        process_dir = output_dir / process
        models = {}
        histories = {}
        for fold, (fold_signal, fold_background) in fold_frames.items():
            models[fold], histories[fold] = _train_fold_model(
                cfg=cfg,
                training_variables=training_variables,
                signal=fold_signal,
                background=fold_background,
                weight_column=weight_column,
                balance_with_absolute_yields=(source_weight is not None),
                squeezing_limit=squeezing_limit,
                device=device,
                checkpoint_dir=process_dir / fold,
                fold_label=fold,
            )
            save_model(models[fold], process_dir / fold)
            save_loss_history(histories[fold], process_dir / fold)

        combined = FoldCombinedDNN(
            even_model=models["fold_even"],
            odd_model=models["fold_odd"],
            fold_id_name="event",
        )
        save_model(combined, process_dir)
        save_loss_history(
            histories["fold_even"] + histories["fold_odd"],
            process_dir,
        )

    metadata = {
        "squeezing_probability": squeezing,
        "reduced_weight_grouping": reduced_weight_grouping,
        "model_type": "DNN",
        "grouping": None,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return output_dir
