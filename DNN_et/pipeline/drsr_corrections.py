from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch as t

from classes import LikelihoodRatioCalculation, load_data, load_model
from classes.DataHandling import FeatureRegistry, FeatureStore
from groupings import GROUPING_NAMES, grouping_suffix, squeezing_feature_suffix
from training_fraction_fake_factors import (
    FRACTION_COLUMNS,
    PROCESSES,
    process_fake_factor_columns,
)


def drsr_correction_feature_suffix(squeezing: Optional[float] = 0.99) -> str:
    return squeezing_feature_suffix(squeezing)


def drsr_correction_name(
    process: str,
    squeezing: Optional[float] = 0.99,
) -> str:
    if process not in PROCESSES:
        raise ValueError(f"Unsupported DRSR correction process: {process}")
    return f"correction_drsr_{process}{drsr_correction_feature_suffix(squeezing)}"


def drsr_corrected_training_fraction_fake_factor_name(
    grouping: str = "njets",
    squeezing: Optional[float] = 0.99,
) -> str:
    if grouping not in GROUPING_NAMES:
        raise ValueError(f"Unsupported FF grouping: {grouping}")
    return (
        "ff_dnn_mlf_drsr_corrected"
        f"{grouping_suffix(grouping)}"
        f"{squeezing_feature_suffix(squeezing)}"
    )


def _load_variables(path: Union[str, Path]):
    import yaml

    with open(path, "r", encoding="utf-8") as stream:
        return (yaml.safe_load(stream) or {}).get("variables", [])


def _prepare_model_input(model, frame: pd.DataFrame, training_variables, device):
    input_names = getattr(model, "_input_names", None)
    features = list(input_names) if input_names is not None else list(training_variables)
    features = [
        name for name in features
        if name not in ("event", "event_parity")
    ]
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise KeyError(f"DRSR model input is missing columns: {missing}")
    return t.from_numpy(frame[features].to_numpy(dtype=np.float32)).to(device)


def _predict_drsr_correction(
    *,
    model,
    frame: pd.DataFrame,
    training_variables,
    device,
    batch_size: int,
) -> np.ndarray:
    ratio_model = LikelihoodRatioCalculation(
        model,
        normalization_constants=1.0,
        clip=(1.0e-8, float("inf")),
    ).to(device).eval()

    corrections = np.full(len(frame), np.nan, dtype=np.float32)
    with t.inference_mode():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            batch = _prepare_model_input(
                ratio_model,
                frame.iloc[start:stop],
                training_variables,
                device,
            )
            prediction = ratio_model(batch).detach().cpu().reshape(-1).numpy()
            corrections[start:stop] = prediction.astype(np.float32)

    if not np.isfinite(corrections).all():
        invalid = int((~np.isfinite(corrections)).sum())
        raise ValueError(
            f"DRSR correction prediction produced {invalid}/{len(frame)} "
            "non-finite values."
        )
    return corrections


def _load_feature_file(df, path: Union[str, Path]) -> None:
    path = Path(path)
    feature_frame = pd.read_feather(path)
    key_column = "row_index" if "row_index" in feature_frame.columns else "event"
    if key_column not in feature_frame.columns:
        raise KeyError(
            f"Feature file {path} does not contain a 'row_index' or 'event' column."
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
        if key_column == "row_index":
            df.events[column] = df.events.index.to_series().map(compact[column])
        else:
            df.events[column] = df.events["event"].map(compact[column])


def calculate_and_store_drsr_correction_factors(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    training_variables_path: Union[str, Path],
    drsr_model_dir: Union[str, Path],
    feature_store_path: Union[str, Path],
    feature_registry_path: Union[str, Path],
    squeezing: Optional[float] = 0.99,
    batch_size: int = 65536,
) -> Path:
    """Calculate C(x)=NN(x)/(1-NN(x)) for each process on all AR rows."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    df = load_data(data_path, masks_path)
    frame = df.AR.events.copy()
    if frame.empty:
        raise ValueError("No events selected in AR for DRSR corrections.")

    training_variables = _load_variables(training_variables_path)
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    drsr_model_dir = Path(drsr_model_dir)

    feature_df = pd.DataFrame({
        "row_index": frame.index.to_numpy(dtype=np.int64),
        "event": frame["event"].to_numpy(),
    })
    for process in PROCESSES:
        model_dir = drsr_model_dir / process
        if not (model_dir / "model_weights.pth").is_file():
            raise FileNotFoundError(f"Missing DRSR model for {process}: {model_dir}")
        model = load_model(model_dir, device=str(device)).eval()
        column = drsr_correction_name(process, squeezing=squeezing)
        feature_df[column] = _predict_drsr_correction(
            model=model,
            frame=frame,
            training_variables=training_variables,
            device=device,
            batch_size=batch_size,
        )

    registry = FeatureRegistry(feature_registry_path)
    store = FeatureStore(feature_store_path, registry)
    store.write(feature_df)
    store.save()
    registry.save()
    return Path(feature_store_path)


def calculate_drsr_corrected_training_fraction_fake_factor(
    frame: pd.DataFrame,
    *,
    grouping: str = "njets",
    squeezing: Optional[float] = 0.99,
) -> str:
    """Combine process FFs with NN fractions and DRSR process corrections."""
    output_name = drsr_corrected_training_fraction_fake_factor_name(
        grouping=grouping,
        squeezing=squeezing,
    )
    process_columns = process_fake_factor_columns(
        grouping=grouping,
        squeezing=squeezing,
    )
    correction_columns = {
        process: drsr_correction_name(process, squeezing=squeezing)
        for process in PROCESSES
    }
    required_columns = {
        *FRACTION_COLUMNS.values(),
        *process_columns.values(),
        *correction_columns.values(),
    }
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise KeyError(
            f"Missing columns for {output_name} calculation: {missing}"
        )

    finite = np.isfinite(
        frame.loc[:, sorted(required_columns)].to_numpy(dtype=np.float64)
    ).all(axis=1)
    if not finite.all():
        invalid = int((~finite).sum())
        nonfinite_by_column = {
            column: int((~np.isfinite(
                frame[column].to_numpy(dtype=np.float64)
            )).sum())
            for column in sorted(required_columns)
        }
        raise ValueError(
            f"{output_name} has {invalid}/{len(frame)} AR events with "
            "non-finite fraction, fake-factor, or DRSR correction inputs. "
            f"Non-finite counts by column: {nonfinite_by_column}"
        )

    fraction_sum = sum(
        frame[FRACTION_COLUMNS[process]].to_numpy(dtype=np.float64)
        for process in PROCESSES
    )
    if not np.allclose(fraction_sum, 1.0, rtol=0.0, atol=5e-4):
        invalid = int((np.abs(fraction_sum - 1.0) > 5e-4).sum())
        raise ValueError(
            f"{invalid}/{len(frame)} NN fraction sums differ from one by "
            f"more than 5e-4 for {output_name}."
        )

    frame[output_name] = sum(
        frame[FRACTION_COLUMNS[process]].to_numpy(dtype=np.float64)
        * frame[process_columns[process]].to_numpy(dtype=np.float64)
        * frame[correction_columns[process]].to_numpy(dtype=np.float64)
        for process in PROCESSES
    )
    return output_name


def calculate_and_store_drsr_corrected_training_fraction_fake_factors(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    fraction_feature_path: Union[str, Path],
    fake_factor_feature_path: Union[str, Path],
    drsr_correction_feature_path: Union[str, Path],
    feature_store_path: Union[str, Path],
    feature_registry_path: Union[str, Path],
    grouping: str = "njets",
    squeezing: Optional[float] = 0.99,
) -> Path:
    """Calculate DRSR-corrected MLF fake factors for all events in AR."""
    for path, description in (
        (fraction_feature_path, "training-fraction feature file"),
        (fake_factor_feature_path, "process fake-factor feature file"),
        (drsr_correction_feature_path, "DRSR correction feature file"),
    ):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")

    df = load_data(data_path, masks_path)
    _load_feature_file(df, fraction_feature_path)
    _load_feature_file(df, fake_factor_feature_path)
    _load_feature_file(df, drsr_correction_feature_path)

    calculation_frame = df.AR.events.copy()
    if calculation_frame.empty:
        raise ValueError(
            "No events selected in AR for DRSR-corrected MLF fake factors."
        )

    output_name = calculate_drsr_corrected_training_fraction_fake_factor(
        calculation_frame,
        grouping=grouping,
        squeezing=squeezing,
    )
    output_values = calculation_frame[output_name].to_numpy(dtype=np.float64)
    if not np.isfinite(output_values).all():
        invalid = int((~np.isfinite(output_values)).sum())
        raise ValueError(
            f"{output_name} calculation produced {invalid} non-finite values "
            "in AR."
        )

    feature_df = pd.DataFrame({
        "row_index": calculation_frame.index.to_numpy(dtype=np.int64),
        "event": calculation_frame["event"].to_numpy(),
        output_name: output_values.astype(np.float32),
    })
    registry = FeatureRegistry(feature_registry_path)
    store = FeatureStore(feature_store_path, registry)
    store.write(feature_df)
    store.save()
    registry.save()
    return Path(feature_store_path)
