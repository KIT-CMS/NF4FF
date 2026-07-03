import torch as t
import numpy as np
import random
import logging
import yaml
import json
import argparse

import pandas as pd
from dataclasses import dataclass, fields, is_dataclass
from typing import Callable, Iterable, Tuple, Union
from classes import (
    FoldCombinedDNN,
    LikelihoodRatioCalculation,
    create_training_dataset,
    DNN,
    load_data,
    load_model,
    save_model,
    train_dnn_squeezed_loss,
)
from pathlib import Path

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
OUTPUT_DIR = (
    PROJECT_ROOT / "Law_workflow_results" / "DRSR_models_squeezed" / "0.9900"
)
GROUPED_DNN_COMBINED_MODEL_DIR = (
    PROJECT_ROOT / "Law_workflow_results" / "CombinedModels" / "0.9900"
)
QCD_EXTRAPOLATION_FEATURE_PATH = (
    PROJECT_ROOT
    / "Law_workflow_results"
    / "data"
    / "features"
    / "qcd_extrapolation"
    / "qcd_extrapolation_weights_njets.feather"
)
PROCESS_OUTPUT_NAMES = {
    "wjets": "Wjets",
    "qcd": "QCD",
    "ttbar": "ttbar",
}


def drsr_loss_limit_label(squeezing_loss_limit: float) -> str:
    value = str(float(squeezing_loss_limit)).replace(".", "p").replace("-", "m")
    return f"loss_squeeze_pm{value}"


def drsr_output_dir(
    squeezing_loss_limit: float,
    base_output_dir: Union[str, Path] = OUTPUT_DIR,
) -> Path:
    return Path(base_output_dir) / drsr_loss_limit_label(squeezing_loss_limit)


@dataclass(frozen=True)
class ProcessConfig:
    name: str
    signal_getter: Callable
    background_getter: Callable
    model_process_name: str
    input_weight_column: str = "weight"
    output_weight_column: str = "weight"
    feature_file: Union[Path, None] = None
    feature_key_column: str = "row_index"


def process_model_path(
    process_config: ProcessConfig,
    combined_model_dir: Union[str, Path],
) -> Path:
    return (
        Path(combined_model_dir)
        / PROCESS_OUTPUT_NAMES[process_config.model_process_name]
        / "njets"
        / "torch_model"
        / "model_full.dill"
    )


def default_process_configs(
    qcd_extrapolation_feature_path: Union[str, Path] = (
        QCD_EXTRAPOLATION_FEATURE_PATH
    ),
) -> Tuple[ProcessConfig, ...]:
    return (
        ProcessConfig(
            name="wjets",
            signal_getter=lambda df: df.wjets.SR.events,
            background_getter=lambda df: df.wjets.AR.events,
            model_process_name="wjets",
        ),
        ProcessConfig(
            name="qcd",
            signal_getter=lambda df: df.data.SR_SS.events,
            background_getter=lambda df: df.data.AR_SS.events,
            model_process_name="qcd",
            input_weight_column="weight_qcd_extrapolation_njets",
            output_weight_column="weight",
            feature_file=Path(qcd_extrapolation_feature_path),
            feature_key_column="event",
        ),
        ProcessConfig(
            name="ttbar",
            signal_getter=lambda df: df.ttbar.SR.events,
            background_getter=lambda df: df.ttbar.AR.events,
            model_process_name="ttbar",
        ),
    )


@dataclass(frozen=True)
class ModelConfig:
    hidden_nodes: Tuple[int, ...]
    output_nodes: int
    activation: str = "ReLU"
    output_activation: str = "Sigmoid"
    dropout: Union[float, Tuple[float, ...]] = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    lr: float = 1e-3
    loss: str = "BCE"


@dataclass(frozen=True)
class SchedulerConfig:
    patience: int = 10
    early_stopping_patience: int = 20
    factor: float = 0.1
    min_delta: float = 1.0e-4
    min_lr: float = 1e-6


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    training: TrainingConfig
    scheduler: SchedulerConfig


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


def load_variables(path):
    with open(path, "r") as stream:
        return (yaml.safe_load(stream) or {}).get("variables", [])


def _load_model_from_path(path: Path, device: str):
    """load_model expects a model directory; accept model_full.dill paths too."""
    path = Path(path)
    model_dir = path.parent if path.name == "model_full.dill" else path
    return load_model(model_dir, device=device).eval()


def _model_feature_names(model, fallback_features):
    input_names = getattr(model, "_input_names", None)
    if input_names is None:
        return list(fallback_features)

    input_names = list(input_names)
    fold_id_name = getattr(model, "_fold_id_name", "event")
    if isinstance(model, FoldCombinedDNN) and input_names:
        if input_names[0] == fold_id_name:
            return input_names[1:]
    return [
        name for name in input_names
        if name not in ("event", "event_parity", fold_id_name)
    ]


def _requires_fold_input(model) -> bool:
    if isinstance(model, FoldCombinedDNN):
        return True
    wrapped = getattr(model, "wrapped_model", None)
    if wrapped is not None:
        return _requires_fold_input(wrapped)
    return False


def _prepare_model_input(model, frame, training_variables, device):
    features = _model_feature_names(model, training_variables)
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise KeyError(f"Model input is missing columns: {missing}")

    X = t.from_numpy(frame[features].to_numpy(dtype=np.float32)).to(device)
    if _requires_fold_input(model):
        event_ids = t.from_numpy(
            frame["event"].to_numpy(dtype=np.float32)
        ).to(device)
        return t.cat([event_ids.unsqueeze(0), X.T], dim=0)
    return X


def _normalization_constant(signal, background, weight_column="weight"):
    numerator = float(signal[weight_column].sum())
    denominator = float(background[weight_column].sum())
    if not np.isfinite(denominator) or denominator == 0.0:
        raise ValueError(
            f"Cannot calculate fake-factor normalization: "
            f"background yield is {denominator}."
        )
    return numerator / denominator


def _model_returns_fake_factors(model) -> bool:
    if isinstance(model, LikelihoodRatioCalculation):
        return True
    if isinstance(model, FoldCombinedDNN):
        return (
            isinstance(model.even_model, LikelihoodRatioCalculation)
            and isinstance(model.odd_model, LikelihoodRatioCalculation)
        )
    wrapped = getattr(model, "wrapped_model", None)
    if wrapped is not None:
        return _model_returns_fake_factors(wrapped)
    return False


def predict_fake_factors(
    model,
    frame,
    training_variables,
    signal,
    background,
    device,
    weight_column="weight",
):
    X = _prepare_model_input(model, frame, training_variables, device)
    with t.inference_mode():
        output = model(X).detach().cpu().reshape(-1).numpy()

    if _model_returns_fake_factors(model):
        fake_factors = output
    else:
        eps = 1e-6
        probabilities = np.clip(output, eps, 1.0 - eps)
        ratio = probabilities / (1.0 - probabilities)
        fake_factors = ratio * _normalization_constant(
            signal=signal,
            background=background,
            weight_column=weight_column,
        )

    fake_factors = np.clip(fake_factors, 0.0, 10.0)
    if len(fake_factors) != len(frame):
        raise ValueError(
            "Model output length does not match application frame length: "
            f"{len(fake_factors)} vs {len(frame)}."
        )
    return fake_factors


def build_fake_factor_weighted_background(
    model,
    signal,
    background,
    training_variables,
    device,
    weight_column="weight",
    output_weight_column="weight",
):
    background = background.copy()
    if weight_column not in background.columns:
        raise KeyError(
            f"Background is missing input weight column {weight_column!r}."
        )
    fake_factors = predict_fake_factors(
        model=model,
        frame=background,
        training_variables=training_variables,
        signal=signal,
        background=background,
        device=device,
        weight_column=weight_column,
    )
    background["fake_factor"] = fake_factors
    background[output_weight_column] = (
        background[weight_column].to_numpy(dtype=np.float64)
        * fake_factors
    )
    return background


def _load_feature_file(
    df,
    path: Union[str, Path],
    key_column: str,
):
    path = Path(path)
    if key_column == "row_index":
        df.load_feature_file(path)
        return

    feature_frame = pd.read_feather(path)
    if key_column not in feature_frame.columns:
        raise KeyError(
            f"Feature file {path} does not contain key column "
            f"{key_column!r}."
        )
    feature_columns = [
        column for column in feature_frame.columns
        if column not in ("event", "row_index")
    ]
    if not feature_columns:
        return

    compact = (
        feature_frame[[key_column, *feature_columns]]
        .groupby(key_column, as_index=False, sort=False)
        .last()
        .set_index(key_column)
    )
    for column in feature_columns:
        df.events[column] = df.events[key_column].map(compact[column])


def _require_finite_training_rows(
    frame,
    training_variables,
    weight_column,
    label,
):
    required_columns = [*training_variables, weight_column]
    missing = [
        column for column in required_columns
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"{label}: missing columns {missing}.")

    finite = np.isfinite(
        frame[required_columns].to_numpy(dtype=np.float64)
    ).all(axis=1)
    invalid = int((~finite).sum())
    if invalid:
        weight_values = frame[weight_column].to_numpy(dtype=np.float64)
        nonfinite_weights = int((~np.isfinite(weight_values)).sum())
        raise ValueError(
            f"{label}: {invalid}/{len(frame)} rows have non-finite "
            f"training features or {weight_column}. "
            f"Non-finite weights: {nonfinite_weights}."
        )
    return frame


def _prepare_process_frames(df, process_config: ProcessConfig):
    if process_config.feature_file is not None:
        logger.info(
            "%s: loading feature file %s",
            process_config.name,
            process_config.feature_file,
        )
        _load_feature_file(
            df,
            process_config.feature_file,
            key_column=process_config.feature_key_column,
        )

    signal = process_config.signal_getter(df).copy()
    background = process_config.background_getter(df).copy()
    if process_config.input_weight_column not in background.columns:
        raise KeyError(
            f"{process_config.name}: background is missing "
            f"{process_config.input_weight_column!r}."
        )
    return signal, background


def _train_fold_model(
    *,
    cfg,
    training_variables,
    signal,
    background,
    weight_column,
    balance_with_absolute_yields,
    squeezing_lower_limit,
    squeezing_upper_limit,
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
        squeezing_lower_limit=squeezing_lower_limit,
        squeezing_limit=squeezing_upper_limit,
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


def _write_process_history(history, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "loss_history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )


def train_drsr_squeezed_models(
    *,
    data_path: Union[str, Path] = DATA_PATH,
    masks_path: Union[str, Path] = MASKS_PATH,
    training_var_path: Union[str, Path] = TRAINING_VAR_PATH,
    nn_config_path: Union[str, Path] = NN_CONFIG_PATH,
    combined_model_dir: Union[str, Path] = GROUPED_DNN_COMBINED_MODEL_DIR,
    qcd_extrapolation_feature_path: Union[
        str,
        Path,
    ] = QCD_EXTRAPOLATION_FEATURE_PATH,
    output_dir: Union[str, Path] = OUTPUT_DIR,
    process_configs: Union[Iterable[ProcessConfig], None] = None,
    squeezing_loss_limit: float = 0.1,
):
    if squeezing_loss_limit < 0.0:
        raise ValueError("squeezing_loss_limit must be non-negative.")
    squeezing_lower_limit = -float(squeezing_loss_limit)
    squeezing_upper_limit = float(squeezing_loss_limit)
    device = "cuda" if t.cuda.is_available() else "cpu"

    df = load_data(data_path, masks_path)
    cfg = load_config(nn_config_path, Config)
    training_variables = load_variables(training_var_path)
    output_dir = drsr_output_dir(
        squeezing_loss_limit=squeezing_loss_limit,
        base_output_dir=output_dir,
    )
    combined_model_dir = Path(combined_model_dir)
    if process_configs is None:
        process_configs = default_process_configs(
            qcd_extrapolation_feature_path=qcd_extrapolation_feature_path,
        )
    outputs = {}

    for process_config in process_configs:
        logger.info("Preparing DRSR training for %s", process_config.name)
        signal, background_source = _prepare_process_frames(
            df,
            process_config,
        )
        source_model = _load_model_from_path(
            process_model_path(process_config, combined_model_dir),
            device=device,
        )
        background = build_fake_factor_weighted_background(
            model=source_model,
            signal=signal,
            background=background_source,
            training_variables=training_variables,
            device=device,
            weight_column=process_config.input_weight_column,
            output_weight_column=process_config.output_weight_column,
        )
        signal = _require_finite_training_rows(
            signal,
            training_variables,
            process_config.output_weight_column,
            f"{process_config.name}/signal",
        )
        background = _require_finite_training_rows(
            background,
            training_variables,
            process_config.output_weight_column,
            f"{process_config.name}/background",
        )
        logger.info(
            "%s: built FF-weighted background rows=%d, FF min=%.6g, "
            "FF max=%.6g, weighted yield=%.6g",
            process_config.name,
            len(background),
            float(background["fake_factor"].min()),
            float(background["fake_factor"].max()),
            float(background[process_config.output_weight_column].sum()),
        )

        process_output_dir = output_dir / process_config.name
        model, history = _train_fold_model(
            cfg=cfg,
            training_variables=training_variables,
            signal=signal,
            background=background,
            weight_column=process_config.output_weight_column,
            balance_with_absolute_yields=True,
            squeezing_lower_limit=squeezing_lower_limit,
            squeezing_upper_limit=squeezing_upper_limit,
            device=device,
            checkpoint_dir=process_output_dir,
            fold_label="full_dataset",
        )
        save_model(model, process_output_dir)
        _write_process_history(history, process_output_dir)
        outputs[process_config.name] = {
            "model": model,
            "history": history,
            "output_dir": process_output_dir,
        }

    metadata = {
        "combined_model_dir": str(combined_model_dir),
        "qcd_extrapolation_feature_path": str(qcd_extrapolation_feature_path),
        "squeezing_loss_limit": squeezing_loss_limit,
        "squeezing_lower_limit": squeezing_lower_limit,
        "squeezing_upper_limit": squeezing_upper_limit,
        "processes": [
            {
                "name": config.name,
                "model_process_name": config.model_process_name,
                "model_path": str(process_model_path(
                    config,
                    combined_model_dir,
                )),
                "input_weight_column": config.input_weight_column,
                "output_weight_column": config.output_weight_column,
                "feature_file": (
                    None if config.feature_file is None
                    else str(config.feature_file)
                ),
                "feature_key_column": config.feature_key_column,
            }
            for config in process_configs
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train DRSR squeezed-loss models with symmetric loss squeezing."
        )
    )
    parser.add_argument(
        "squeezing_loss_limit",
        nargs="?",
        type=float,
        default=0.1,
        help=(
            "Symmetric loss squeezing limit. For example, 0.1 uses lower "
            "limit -0.1 and upper limit +0.1."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return train_drsr_squeezed_models(
        squeezing_loss_limit=args.squeezing_loss_limit,
    )


if __name__ == "__main__":
    main()
