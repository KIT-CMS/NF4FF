from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from groupings import (
    GROUPING_NAMES,
    grouping_source,
    grouping_suffix,
    squeezing_feature_suffix,
)


PROCESSES = {
    "wjets": "Wjets",
    "qcd": "QCD",
    "ttbar": "ttbar",
}


def _squeezing_directory_label(squeezing: Optional[float]) -> str:
    return "unsqueezed" if squeezing is None else f"{squeezing:.4f}"


def evaluate_compound_ff_correction(
    correction_set,
    compound_name: str,
    df: pd.DataFrame,
) -> np.ndarray:
    """Evaluate one compound FF correction for all rows in ``df``."""
    compound_correction = correction_set.compound[compound_name]
    expected_inputs = [
        input_spec.name for input_spec in compound_correction.inputs
    ]
    missing_inputs = sorted(
        name
        for name in expected_inputs
        if name != "syst" and name not in df.columns
    )
    if missing_inputs:
        raise KeyError(
            f"Missing inputs for correction {compound_name}: "
            f"{missing_inputs}"
        )

    ordered_inputs = [
        "nominal" if name == "syst" else df[name].to_numpy()
        for name in expected_inputs
    ]
    return np.asarray(
        compound_correction.evaluate(*ordered_inputs),
        dtype=np.float64,
    )


def _load_correction_sets(correction_dir: Path):
    try:
        import correctionlib as cr
    except ImportError as error:
        raise ImportError(
            "Corrected fake-factor calculation requires correctionlib."
        ) from error

    fake_factors_path = correction_dir / "fake_factors_et.json.gz"
    corrections_path = correction_dir / "FF_corrections_et.json.gz"
    missing_paths = [
        path
        for path in (fake_factors_path, corrections_path)
        if not path.is_file()
    ]
    if missing_paths:
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing corrected-FF correctionlib files:\n"
            f"{missing}\n"
            "Set --correction-set-root to the directory containing the "
            "<squeezing>/<grouping> hierarchy."
        )

    return (
        cr.CorrectionSet.from_file(str(fake_factors_path)),
        cr.CorrectionSet.from_file(str(corrections_path)),
    )


def calculate_corrected_fake_factors(
    df: pd.DataFrame,
    correction_set_root: Union[str, Path],
    *,
    grouping: str = "tau_decaymode_2",
    squeezing: Optional[float] = None,
) -> str:
    """
    Calculate corrected DNN fake factors and add them to ``df``.

    The correction files are read from
    ``<correction_set_root>/<squeezing>/<grouping>``.

    Returns:
        Name of the column added to ``df``.
    """
    if grouping not in GROUPING_NAMES:
        raise ValueError(f"Unsupported corrected FF grouping: {grouping}")

    feature_suffix = (
        f"{grouping_suffix(grouping)}"
        f"{squeezing_feature_suffix(squeezing)}"
    )
    process_columns = {
        process: f"ff_dnn_{process}{feature_suffix}"
        for process in PROCESSES
    }
    required_columns = {
        "mt_1",
        "njets",
        grouping_source(grouping),
        *process_columns.values(),
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise KeyError(
            "Missing columns for corrected fake-factor calculation: "
            f"{missing_columns}"
        )

    correction_dir = (
        Path(correction_set_root)
        / _squeezing_directory_label(squeezing)
        / grouping
    )
    fake_factors, corrections = _load_correction_sets(correction_dir)
    process_fractions = fake_factors["process_fractions"]

    finite = np.isfinite(
        df.loc[:, sorted(required_columns)].to_numpy(dtype=np.float64)
    ).all(axis=1)
    output = np.full(len(df), np.nan, dtype=np.float64)
    if not finite.any():
        raise ValueError(
            "No rows have finite inputs for corrected fake-factor "
            "calculation."
        )

    finite_frame = df.loc[finite]
    corrected_process_factors = {}
    fractions = {}
    for process, correction_name in PROCESSES.items():
        fractions[process] = np.asarray(
            process_fractions.evaluate(
                correction_name,
                finite_frame["mt_1"].to_numpy(),
                finite_frame["njets"].to_numpy(),
                "nominal",
            ),
            dtype=np.float64,
        )
        correction = evaluate_compound_ff_correction(
            corrections,
            f"{correction_name}_compound_correction",
            finite_frame,
        )
        corrected_process_factors[process] = (
            finite_frame[process_columns[process]].to_numpy(
                dtype=np.float64
            )
            * correction
        )

    output[finite] = sum(
        fractions[process] * corrected_process_factors[process]
        for process in PROCESSES
    )
    output_name = f"ff_dnn_corrected{feature_suffix}"
    df[output_name] = output
    return output_name


def calculate_and_store_corrected_fake_factors(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    fake_factor_feature_path: Union[str, Path],
    correction_set_root: Union[str, Path],
    feature_store_path: Union[str, Path],
    feature_registry_path: Union[str, Path],
    groupings=GROUPING_NAMES,
    squeezing: Optional[float] = None,
) -> Path:
    """Calculate corrected DNN fake factors in AR and store them as features."""
    from classes import load_data
    from classes.DataHandling import FeatureRegistry, FeatureStore

    fake_factor_feature_path = Path(fake_factor_feature_path)
    if not fake_factor_feature_path.is_file():
        raise FileNotFoundError(
            "Missing DNN fake-factor feature file: "
            f"{fake_factor_feature_path}"
        )

    df = load_data(data_path, masks_path)
    df.load_feature_file(fake_factor_feature_path)
    calculation_frame = df.AR.events.copy()
    if calculation_frame.empty:
        raise ValueError(
            "No events selected in AR for corrected fake-factor calculation."
        )

    feature_df = pd.DataFrame({
        "row_index": calculation_frame.index,
        "event": calculation_frame["event"].to_numpy(),
    })
    for grouping in tuple(groupings):
        output_name = calculate_corrected_fake_factors(
            calculation_frame,
            correction_set_root,
            grouping=grouping,
            squeezing=squeezing,
        )
        feature_values = calculation_frame[output_name].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(feature_values).all():
            invalid_count = int((~np.isfinite(feature_values)).sum())
            raise ValueError(
                f"{output_name} calculation produced "
                f"{invalid_count} non-finite values in AR."
            )
        feature_df[output_name] = feature_values

    registry = FeatureRegistry(feature_registry_path)
    store = FeatureStore(feature_store_path, registry)
    store.write(feature_df)
    store.save()
    registry.save()
    return Path(feature_store_path)
