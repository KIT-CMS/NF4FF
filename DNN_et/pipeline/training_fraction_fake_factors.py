from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from classes import load_data
from classes.DataHandling import FeatureRegistry, FeatureStore
from calculate_ff_corrected import (
    PROCESSES as CORRECTIONLIB_PROCESSES,
    _load_correction_sets,
    _squeezing_directory_label,
    evaluate_compound_ff_correction,
)
from groupings import GROUPING_NAMES, grouping_suffix, squeezing_feature_suffix


PROCESSES = ("wjets", "qcd", "ttbar")
FRACTION_COLUMNS = {
    "wjets": "fraction_wjets",
    "qcd": "fraction_qcd",
    "ttbar": "fraction_ttbar",
}


def training_fraction_feature_suffix(
    grouping: str,
    squeezing: Optional[float],
) -> str:
    if grouping not in GROUPING_NAMES:
        raise ValueError(f"Unsupported FF grouping: {grouping}")
    return f"{grouping_suffix(grouping)}{squeezing_feature_suffix(squeezing)}"


def training_fraction_fake_factor_name(
    grouping: str = "njets",
    squeezing: Optional[float] = 0.99,
) -> str:
    return (
        "ff_dnn_mlf"
        f"{training_fraction_feature_suffix(grouping, squeezing)}"
    )


def corrected_training_fraction_fake_factor_name(
    grouping: str = "njets",
    squeezing: Optional[float] = 0.99,
) -> str:
    return (
        "ff_dnn_mlf_corrected"
        f"{training_fraction_feature_suffix(grouping, squeezing)}"
    )


def process_fake_factor_columns(
    grouping: str = "njets",
    squeezing: Optional[float] = 0.99,
) -> dict:
    suffix = training_fraction_feature_suffix(grouping, squeezing)
    return {
        process: f"ff_dnn_{process}{suffix}"
        for process in PROCESSES
    }


def calculate_training_fraction_fake_factor(
    frame: pd.DataFrame,
    *,
    grouping: str = "njets",
    squeezing: Optional[float] = 0.99,
) -> str:
    """
    Combine process DNN fake factors with NN-trained process fractions.

    The input frame is expected to contain the AR events to which the fake
    factor should be applied, including the MC rows needed for closure
    subtraction.
    """
    output_name = training_fraction_fake_factor_name(
        grouping=grouping,
        squeezing=squeezing,
    )
    process_columns = process_fake_factor_columns(
        grouping=grouping,
        squeezing=squeezing,
    )
    required_columns = {
        *FRACTION_COLUMNS.values(),
        *process_columns.values(),
    }
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise KeyError(
            f"Missing columns for {output_name} calculation: "
            f"{missing_columns}"
        )

    calculation_columns = [
        FRACTION_COLUMNS["wjets"],
        FRACTION_COLUMNS["qcd"],
        FRACTION_COLUMNS["ttbar"],
        process_columns["wjets"],
        process_columns["qcd"],
        process_columns["ttbar"],
    ]
    values = frame.loc[:, calculation_columns].to_numpy(dtype=np.float64)
    finite = np.isfinite(values).all(axis=1)
    if not finite.all():
        invalid_count = int((~finite).sum())
        raise ValueError(
            f"{output_name} has {invalid_count}/{len(frame)} AR events "
            "with non-finite fraction or process fake-factor inputs."
        )

    fraction_sum = (
        frame[FRACTION_COLUMNS["wjets"]].to_numpy(dtype=np.float64)
        + frame[FRACTION_COLUMNS["qcd"]].to_numpy(dtype=np.float64)
        + frame[FRACTION_COLUMNS["ttbar"]].to_numpy(dtype=np.float64)
    )
    if not np.allclose(fraction_sum, 1.0, rtol=0.0, atol=5e-4):
        invalid_count = int((np.abs(fraction_sum - 1.0) > 5e-4).sum())
        raise ValueError(
            f"{invalid_count}/{len(frame)} NN fraction sums differ from one "
            f"by more than 5e-4 for {output_name}."
        )

    frame[output_name] = (
        frame[FRACTION_COLUMNS["wjets"]].to_numpy(dtype=np.float64)
        * frame[process_columns["wjets"]].to_numpy(dtype=np.float64)
        + frame[FRACTION_COLUMNS["qcd"]].to_numpy(dtype=np.float64)
        * frame[process_columns["qcd"]].to_numpy(dtype=np.float64)
        + frame[FRACTION_COLUMNS["ttbar"]].to_numpy(dtype=np.float64)
        * frame[process_columns["ttbar"]].to_numpy(dtype=np.float64)
    )
    return output_name


def calculate_corrected_training_fraction_fake_factor(
    frame: pd.DataFrame,
    correction_set_root: Union[str, Path],
    *,
    grouping: str = "njets",
    squeezing: Optional[float] = 0.99,
) -> str:
    """Apply process non-closure corrections and combine with MLF fractions."""
    output_name = corrected_training_fraction_fake_factor_name(
        grouping=grouping,
        squeezing=squeezing,
    )
    process_columns = process_fake_factor_columns(
        grouping=grouping,
        squeezing=squeezing,
    )
    required_columns = {
        *FRACTION_COLUMNS.values(),
        *process_columns.values(),
    }
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise KeyError(
            f"Missing columns for {output_name} calculation: "
            f"{missing_columns}"
        )

    correction_dir = (
        Path(correction_set_root)
        / _squeezing_directory_label(squeezing)
        / grouping
    )
    _, corrections = _load_correction_sets(correction_dir)

    finite_columns = [
        *FRACTION_COLUMNS.values(),
        *process_columns.values(),
    ]
    finite = np.isfinite(
        frame.loc[:, finite_columns].to_numpy(dtype=np.float64)
    ).all(axis=1)
    if not finite.all():
        invalid_count = int((~finite).sum())
        raise ValueError(
            f"{output_name} has {invalid_count}/{len(frame)} AR events "
            "with non-finite fraction or process fake-factor inputs."
        )

    finite_frame = frame.loc[finite]
    output = np.full(len(frame), np.nan, dtype=np.float64)
    output[finite] = sum(
        finite_frame[FRACTION_COLUMNS[process]].to_numpy(dtype=np.float64)
        * finite_frame[process_columns[process]].to_numpy(dtype=np.float64)
        * evaluate_compound_ff_correction(
            corrections,
            f"{correction_name}_compound_correction",
            finite_frame,
        )
        for process, correction_name in CORRECTIONLIB_PROCESSES.items()
    )
    frame[output_name] = output
    return output_name


def calculate_and_store_training_fraction_fake_factors(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    fraction_feature_path: Union[str, Path],
    fake_factor_feature_path: Union[str, Path],
    feature_store_path: Union[str, Path],
    feature_registry_path: Union[str, Path],
    grouping: str = "njets",
    squeezing: Optional[float] = 0.99,
) -> Path:
    """Calculate training-fraction-combined fake factors in AR."""
    fraction_feature_path = Path(fraction_feature_path)
    fake_factor_feature_path = Path(fake_factor_feature_path)
    feature_store_path = Path(feature_store_path)
    feature_registry_path = Path(feature_registry_path)

    for path, description in (
        (fraction_feature_path, "training-fraction feature file"),
        (fake_factor_feature_path, "process fake-factor feature file"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")

    df = load_data(data_path, masks_path)
    df.load_feature_file(fraction_feature_path)
    df.load_feature_file(fake_factor_feature_path)

    calculation_frame = df.AR.events.copy()
    if calculation_frame.empty:
        raise ValueError(
            "No events selected in AR for training-fraction fake factors."
        )

    output_name = calculate_training_fraction_fake_factor(
        calculation_frame,
        grouping=grouping,
        squeezing=squeezing,
    )
    output_values = calculation_frame[output_name].to_numpy(dtype=np.float64)
    if not np.isfinite(output_values).all():
        invalid_count = int((~np.isfinite(output_values)).sum())
        raise ValueError(
            f"{output_name} calculation produced {invalid_count} "
            "non-finite values in AR."
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
    return feature_store_path


def calculate_and_store_corrected_training_fraction_fake_factors(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    fraction_feature_path: Union[str, Path],
    fake_factor_feature_path: Union[str, Path],
    correction_set_root: Union[str, Path],
    feature_store_path: Union[str, Path],
    feature_registry_path: Union[str, Path],
    grouping: str = "njets",
    squeezing: Optional[float] = 0.99,
) -> Path:
    """Calculate non-closure-corrected MLF fake factors in AR."""
    fraction_feature_path = Path(fraction_feature_path)
    fake_factor_feature_path = Path(fake_factor_feature_path)
    feature_store_path = Path(feature_store_path)
    feature_registry_path = Path(feature_registry_path)

    for path, description in (
        (fraction_feature_path, "training-fraction feature file"),
        (fake_factor_feature_path, "process fake-factor feature file"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")

    df = load_data(data_path, masks_path)
    df.load_feature_file(fraction_feature_path)
    df.load_feature_file(fake_factor_feature_path)

    calculation_frame = df.AR.events.copy()
    if calculation_frame.empty:
        raise ValueError(
            "No events selected in AR for corrected MLF fake factors."
        )

    output_name = calculate_corrected_training_fraction_fake_factor(
        calculation_frame,
        correction_set_root,
        grouping=grouping,
        squeezing=squeezing,
    )
    output_values = calculation_frame[output_name].to_numpy(dtype=np.float64)
    if not np.isfinite(output_values).all():
        invalid_count = int((~np.isfinite(output_values)).sum())
        raise ValueError(
            f"{output_name} calculation produced {invalid_count} "
            "non-finite values in AR."
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
    return feature_store_path
