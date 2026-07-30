"""Grouped fake-factor model training."""

import argparse
import torch as t
import numpy as np
import random
import logging
import yaml
import csv
import json
from data.handling import create_training_dataset, load_data
from models.networks import DNN, FoldCombinedDNN, GroupedDNN, save_model
from training.engine import train_dnn_squeezed_loss
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Dict, Any
from dataclasses import is_dataclass, fields
from pathlib import Path
from core.groupings import GROUPING_NAMES, grouping_bounds, grouping_source
from core.paths import CONFIG_ROOT, PROJECT_ROOT

SEED = 42
logger = logging.getLogger(__name__)


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

DATA_PATH = PROJECT_ROOT / 'data' / 'dataframe_complete.feather'
MASKS_PATH = CONFIG_ROOT / 'selections.yaml'
TRAINING_VAR_PATH = CONFIG_ROOT / 'variables_fake_factor.yaml'
NN_CONFIG_PATH = CONFIG_ROOT / 'model_fake_factor.yaml'
CHECKPOINT_DIR = PROJECT_ROOT / 'Training_results_squeezed'
REDUCED_WEIGHT_DIR = (
    PROJECT_ROOT / 'data' / 'features' / 'reduced_dataset'
)



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


def save_loss_history(history: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "loss_history.json"
    csv_path = output_dir / "loss_history.csv"

    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    if history:
        fieldnames = list(history[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    logger.info("Saved loss history to %s and %s", csv_path, json_path)

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


def squeezing_limit_from_probability(squeezing: Optional[float]) -> Optional[float]:
    if squeezing is None:
        return None
    if not 0.0 < squeezing < 1.0:
        raise ValueError(
            f"squeezing must be between 0 and 1 (exclusive), got {squeezing}"
        )
    return float(np.log(squeezing / (1.0 - squeezing)))


def squeezing_label(squeezing: Optional[float]) -> str:
    return "unsqueezed" if squeezing is None else f"{squeezing:.4f}"


def _prepare_training_frame(
    frame,
    training_variables,
    source_weight,
    weight_column,
    label,
):
    frame = frame.copy()
    frame[weight_column] = frame[source_weight]

    required_columns = [*training_variables, weight_column]
    finite_mask = np.isfinite(
        frame[required_columns].to_numpy(dtype=np.float64)
    ).all(axis=1)
    dropped = int((~finite_mask).sum())
    if dropped:
        logger.warning(
            "%s: dropping %d/%d rows with non-finite features or %s",
            label,
            dropped,
            len(frame),
            source_weight,
        )

    frame = frame.loc[finite_mask].copy()
    if frame.empty:
        raise ValueError(f"{label}: no finite training rows remain.")

    weights = frame[weight_column].to_numpy(dtype=np.float64)
    logger.info(
        "%s uses %s: rows=%d, min=%.6g, max=%.6g, sum=%.6g, abs_sum=%.6g",
        label,
        source_weight,
        len(frame),
        weights.min(),
        weights.max(),
        weights.sum(dtype=np.float64),
        np.abs(weights).sum(dtype=np.float64),
    )
    return frame


def _train_fold_model(
    cfg,
    grouping,
    training_var,
    df_sig,
    df_bkg,
    weight_column,
    balance_column,
    balance_groups,
    balance_with_absolute_yields,
    squeezing_limit,
    device,
    checkpoint_dir,
    fold_label,
):
    train, val = create_training_dataset(
        df_sig=df_sig,
        df_bkg=df_bkg,
        training_var=training_var,
        weight_column=weight_column,
        balance=True,
        balance_column=balance_column,
        balance_groups=balance_groups,
        balance_with_absolute_yields=balance_with_absolute_yields,
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



    model, best_loss, history = train_dnn_squeezed_loss(
        model=model,
        train=train,
        val=val,
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


def train_squeezed_models(
    squeezing: Optional[float] = None,
    data_path: Union[str, Path] = DATA_PATH,
    masks_path: Union[str, Path] = MASKS_PATH,
    training_var_path: Union[str, Path] = TRAINING_VAR_PATH,
    nn_config_path: Union[str, Path] = NN_CONFIG_PATH,
    checkpoint_dir: Union[str, Path] = CHECKPOINT_DIR,
    reduced_weight_dir: Union[str, Path] = REDUCED_WEIGHT_DIR,
):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    cfg = load_config(nn_config_path, Config)
    df = load_data(data_path, masks_path)
    training_var = load_variables(training_var_path)
    checkpoint_dir = Path(checkpoint_dir)
    reduced_weight_dir = Path(reduced_weight_dir)
    penalty_upper_bound = squeezing_limit_from_probability(squeezing)
    output_label = squeezing_label(squeezing)

    taudm_idx = training_var.index('tau_decaymode_2')
    njets_idx = training_var.index('njets')

    logger.info(
        "Squeezing probability=%s, penalty upper bound=%s",
        squeezing,
        penalty_upper_bound,
    )

    grouping_indices = {
        "tau_decaymode_2": taudm_idx,
        "tau_decaymode_2_alt": taudm_idx,
        "njets": njets_idx,
    }
    for group_label in GROUPING_NAMES:
        logger.info('Group splitting: %s', group_label)

        for process in ['wjets', 'qcd', 'ttbar']:
            logger.info('Training process: %s', process)
            group_bounds = grouping_bounds(group_label, process)
            grouping = {
                grouping_indices[group_label]: group_bounds,
            }
            source_column = grouping_source(group_label)

            if process == 'wjets':
                source_weight = f'reduced_weight_wjets_{group_label}_nominal'
                weight_column = 'weight_wjets'
                df.load_feature_file(
                    reduced_weight_dir
                    / 'wjets'
                    / f'reduced_weight_{group_label}.feather'
                )
                df_sig = _prepare_training_frame(
                    df.data.SR_like_wjets.events,
                    training_var,
                    source_weight,
                    weight_column,
                    f"{group_label}/wjets/SR-like",
                )
                df_bkg = _prepare_training_frame(
                    df.data.AR_like_wjets.events,
                    training_var,
                    source_weight,
                    weight_column,
                    f"{group_label}/wjets/AR-like",
                )
            elif process == 'qcd':
                source_weight = f'reduced_weight_qcd_{group_label}_nominal'
                weight_column = 'weight_qcd'
                df.load_feature_file(
                    reduced_weight_dir
                    / 'qcd'
                    / f'reduced_weight_{group_label}.feather'
                )
                df_sig = _prepare_training_frame(
                    df.data.SR_like_qcd.events,
                    training_var,
                    source_weight,
                    weight_column,
                    f"{group_label}/qcd/SR-like",
                )
                df_bkg = _prepare_training_frame(
                    df.data.AR_like_qcd.events,
                    training_var,
                    source_weight,
                    weight_column,
                    f"{group_label}/qcd/AR-like",
                )
            else:
                df_sig = df.ttbar.SR_like_ttbar.events.copy()
                df_bkg = df.ttbar.AR_like_ttbar.events.copy()
                source_weight = None
                weight_column = 'weight'

            df_sig_even = df_sig[df_sig['event'] % 2 == 0]
            df_sig_odd = df_sig[df_sig['event'] % 2 == 1]
            df_bkg_even = df_bkg[df_bkg['event'] % 2 == 0]
            df_bkg_odd = df_bkg[df_bkg['event'] % 2 == 1]

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

            fold_checkpoint_dir = (
                checkpoint_dir / output_label / group_label / process
            )
            even_model, even_history = _train_fold_model(
                cfg=cfg,
                grouping=grouping,
                training_var=training_var,
                df_sig=df_sig_odd,
                df_bkg=df_bkg_odd,
                weight_column=weight_column,
                balance_column=source_column,
                balance_groups=group_bounds,
                balance_with_absolute_yields=(source_weight is not None),
                squeezing_limit=penalty_upper_bound,
                device=device,
                checkpoint_dir=fold_checkpoint_dir / 'fold_even',
                fold_label='fold_even',
            )
            odd_model, odd_history = _train_fold_model(
                cfg=cfg,
                grouping=grouping,
                training_var=training_var,
                df_sig=df_sig_even,
                df_bkg=df_bkg_even,
                weight_column=weight_column,
                balance_column=source_column,
                balance_groups=group_bounds,
                balance_with_absolute_yields=(source_weight is not None),
                squeezing_limit=penalty_upper_bound,
                device=device,
                checkpoint_dir=fold_checkpoint_dir / 'fold_odd',
                fold_label='fold_odd',
            )

            model = FoldCombinedDNN(
                even_model=even_model,
                odd_model=odd_model,
                fold_id_name='event',
            )

            save_model(even_model, fold_checkpoint_dir / 'fold_even')
            save_model(odd_model, fold_checkpoint_dir / 'fold_odd')
            save_model(model, fold_checkpoint_dir)

            save_loss_history(even_history, fold_checkpoint_dir / 'fold_even')
            save_loss_history(odd_history, fold_checkpoint_dir / 'fold_odd')
            save_loss_history(
                even_history + odd_history,
                fold_checkpoint_dir,
            )

    return checkpoint_dir / output_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--squeezing",
        type=float,
        default=None,
        help=(
            "Optional output probability in (0, 1). The loss limit is "
            "computed as ln(squeezing / (1 - squeezing))."
        ),
    )
    args = parser.parse_args()
    train_squeezed_models(squeezing=args.squeezing)


if __name__ == '__main__':
    main()
