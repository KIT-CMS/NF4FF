import logging
from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np
import pandas as pd
import torch as t

from classes import (
    calculate_fake_factor_classic,
    load_data,
    load_model,
    load_variables,
)
from classes.DataHandling import FeatureRegistry, FeatureStore
from groupings import GROUPING_NAMES, grouping_suffix


logger = logging.getLogger(__name__)

DEFAULT_PROCESS_FRACTIONS_PATH = Path(
    "/work/mmoser/TauFakeFactors.back/workdir/"
    "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
)
DEFAULT_CLASSIC_CORRECTIONS_PATH = Path(
    "/work/mmoser/TauFakeFactors.back/workdir/"
    "ff_2026_01_19_check_variable/2018/FF_corrections_et.json.gz"
)
EVALUATION_REGIONS = (
    "SR",
    "AR",
    "SR_like_wjets",
    "AR_like_wjets",
    "SR_like_qcd",
    "AR_like_qcd",
    "SR_like_ttbar",
    "AR_like_ttbar",
)
GROUPINGS = GROUPING_NAMES
PROCESSES = ("wjets", "qcd", "ttbar")
PROCESS_MODEL_DIRS = {
    "wjets": "Wjets",
    "qcd": "QCD",
    "ttbar": "ttbar",
}


def _feature_name(process: str, grouping: str) -> str:
    return f"ff_dnn_{process}{grouping_suffix(grouping)}"


def _application_mask(df, regions: Iterable[str]) -> pd.Series:
    mask = pd.Series(False, index=df.events.index)
    for region in regions:
        if region not in df._manager.regions:
            raise KeyError(f"Unknown fake-factor application region: {region}")
        mask |= df.mask(region)
    return mask


def _model_features(model, configured_features) -> Tuple[str, ...]:
    input_names = getattr(model, "_input_names", None)
    if not input_names or input_names[0] != "event":
        raise ValueError(
            "Combined FF model must declare 'event' as its first input."
        )

    model_features = tuple(input_names[1:])
    configured_features = tuple(configured_features)
    if model_features != configured_features:
        raise ValueError(
            "Combined FF model feature order differs from "
            "training_variables.yaml.\n"
            f"model={model_features}\nconfigured={configured_features}"
        )
    return model_features


def _predict_fake_factors(
    model,
    frame: pd.DataFrame,
    feature_names: Tuple[str, ...],
    *,
    batch_size: int,
    device: t.device,
) -> np.ndarray:
    feature_values = frame.loc[:, feature_names].to_numpy(dtype=np.float32)
    event_values = frame["event"].to_numpy(dtype=np.float32)
    finite = np.isfinite(feature_values).all(axis=1) & np.isfinite(event_values)
    predictions = np.full(len(frame), np.nan, dtype=np.float32)
    valid_indices = np.flatnonzero(finite)
    if len(valid_indices) == 0:
        raise ValueError("No events have finite inputs for FF inference.")

    model = model.eval().to(device)
    with t.no_grad():
        for start in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[start:start + batch_size]
            features = t.as_tensor(
                feature_values[batch_indices],
                dtype=t.float32,
                device=device,
            )
            event_parity = t.as_tensor(
                event_values[batch_indices] % 2,
                dtype=t.float32,
                device=device,
            )
            model_input = t.cat(
                [event_parity.unsqueeze(0), features.T],
                dim=0,
            )
            batch_predictions = (
                model(model_input)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            if len(batch_predictions) != len(batch_indices):
                raise RuntimeError(
                    "FF model returned an unexpected number of predictions: "
                    f"{len(batch_predictions)} for {len(batch_indices)} events."
                )
            predictions[batch_indices] = batch_predictions

    finite_predictions = predictions[finite]
    if not np.isfinite(finite_predictions).all():
        raise ValueError("Combined FF model produced non-finite predictions.")
    return predictions

def calculate_fake_factor_dnn(
    df: pd.DataFrame,
    grouping: str,
    process_fractions,
) -> str:
    """Combine process FFs using the configured process fractions."""
    grouping_columns = {
        "tau_decaymode_2": (
            "ff_dnn",
            ("ff_dnn_wjets", "ff_dnn_qcd", "ff_dnn_ttbar"),
        ),
        "njets": (
            "ff_dnn_njets",
            (
                "ff_dnn_wjets_njets",
                "ff_dnn_qcd_njets",
                "ff_dnn_ttbar_njets",
            ),
        ),
        "tau_decaymode_2_alt": (
            "ff_dnn_tau_decaymode_2_alt",
            (
                "ff_dnn_wjets_tau_decaymode_2_alt",
                "ff_dnn_qcd_tau_decaymode_2_alt",
                "ff_dnn_ttbar_tau_decaymode_2_alt",
            ),
        ),
    }
    if grouping not in grouping_columns:
        raise ValueError(f"Unsupported FF grouping: {grouping}")

    output_name, process_columns = grouping_columns[grouping]
    required_columns = {"mt_1", "njets", *process_columns}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise KeyError(
            f"Missing columns for {output_name} calculation: {missing}"
        )

    finite = np.isfinite(
        df.loc[:, ["mt_1", "njets", *process_columns]].to_numpy(
            dtype=np.float64
        )
    ).all(axis=1)
    output = np.full(len(df), np.nan, dtype=np.float32)
    if not finite.any():
        raise ValueError(f"No finite SR/AR events available for {output_name}.")

    finite_frame = df.loc[finite]
    fractions = {
        "wjets": process_fractions.evaluate(
            "Wjets",
            finite_frame["mt_1"].to_numpy(),
            finite_frame["njets"].to_numpy(),
            "nominal",
        ),
        "qcd": process_fractions.evaluate(
            "QCD",
            finite_frame["mt_1"].to_numpy(),
            finite_frame["njets"].to_numpy(),
            "nominal",
        ),
        "ttbar": process_fractions.evaluate(
            "ttbar",
            finite_frame["mt_1"].to_numpy(),
            finite_frame["njets"].to_numpy(),
            "nominal",
        ),
    }
    output[finite] = (
        fractions["wjets"] * finite_frame[process_columns[0]].to_numpy()
        + fractions["qcd"] * finite_frame[process_columns[1]].to_numpy()
        + fractions["ttbar"] * finite_frame[process_columns[2]].to_numpy()
    )
    df[output_name] = output
    return output_name


def calculate_and_store_fake_factors(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    training_variables_path: Union[str, Path],
    combined_models_dir: Union[str, Path],
    feature_store_path: Union[str, Path],
    feature_registry_path: Union[str, Path],
    process_fractions_path: Union[str, Path] = DEFAULT_PROCESS_FRACTIONS_PATH,
    regions: Iterable[str] = EVALUATION_REGIONS,
    batch_size: int = 65536,
    feature_suffix: str = "",
    groupings: Iterable[str] = GROUPINGS,
) -> Path:
    """Evaluate Law-produced FF models and persist row-index keyed features."""
    combined_models_dir = Path(combined_models_dir)
    feature_store_path = Path(feature_store_path)
    feature_registry_path = Path(feature_registry_path)
    process_fractions_path = Path(process_fractions_path)
    regions = tuple(regions)
    groupings = tuple(groupings)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    unsupported_groupings = sorted(set(groupings).difference(GROUPINGS))
    if unsupported_groupings:
        raise ValueError(f"Unsupported FF groupings: {unsupported_groupings}")

    df = load_data(data_path, masks_path)
    training_variables = tuple(load_variables(training_variables_path))
    for region in regions:
        logger.info(
            "FF application region %s contains %d events.",
            region,
            int(df.mask(region).sum()),
        )
    selected = _application_mask(df, regions)
    inference_frame = df.events.loc[selected].copy()
    if inference_frame.empty:
        raise ValueError("No events selected by the FF application regions.")

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    feature_df = pd.DataFrame({
        "row_index": inference_frame.index,
        "event": inference_frame["event"].to_numpy(),
    })

    for process in PROCESSES:
        for grouping in groupings:
            model_path = (
                combined_models_dir
                / PROCESS_MODEL_DIRS[process]
                / grouping
                / "torch_model"
            )
            if not (model_path / "model_weights.pth").is_file():
                raise FileNotFoundError(
                    f"Missing Law-produced combined FF model: {model_path}"
                )

            model = load_model(model_path, device=str(device))
            model_features = _model_features(model, training_variables)
            output_name = _feature_name(process, grouping)
            feature_df[output_name] = _predict_fake_factors(
                model,
                inference_frame,
                model_features,
                batch_size=batch_size,
                device=device,
            )
            logger.info(
                "Calculated %s for %d events in regions %s "
                "(minimum=%.8g, maximum=%.8g).",
                output_name,
                feature_df[output_name].notna().sum(),
                ", ".join(regions),
                np.nanmin(feature_df[output_name]),
                np.nanmax(feature_df[output_name]),
            )

    if not process_fractions_path.is_file():
        raise FileNotFoundError(
            "Missing process-fraction correction file: "
            f"{process_fractions_path}"
        )
    try:
        import correctionlib as cr
    except ImportError as error:
        raise ImportError(
            "CalculateFakeFactors requires the correctionlib package."
        ) from error
    process_fractions = cr.CorrectionSet.from_file(
        str(process_fractions_path)
    )["process_fractions"]

    feature_values = feature_df.set_index("row_index")
    inference_frame.loc[:, feature_values.columns.difference(["event"])] = (
        feature_values.drop(columns=["event"], errors="ignore")
    )
    sr_ar_mask = df.mask("SR") | df.mask("AR")
    sr_ar_frame = inference_frame.loc[
        inference_frame.index.intersection(df.events.index[sr_ar_mask])
    ].copy()
    if sr_ar_frame.empty:
        raise ValueError("No events selected in the SR or AR regions.")

    for grouping in groupings:
        output_name = calculate_fake_factor_dnn(
            sr_ar_frame,
            grouping,
            process_fractions,
        )
        values_by_row = sr_ar_frame[output_name]
        feature_df[output_name] = feature_df["row_index"].map(values_by_row)
        logger.info(
            "Calculated combined %s for %d SR/AR events "
            "(minimum=%.8g, maximum=%.8g).",
            output_name,
            feature_df[output_name].notna().sum(),
            np.nanmin(feature_df[output_name]),
            np.nanmax(feature_df[output_name]),
        )

    if feature_suffix:
        feature_df = feature_df.rename(columns={
            column: f"{column}{feature_suffix}"
            for column in feature_df.columns
            if column not in ("event", "row_index")
        })

    registry = FeatureRegistry(feature_registry_path)
    store = FeatureStore(feature_store_path, registry)
    store.write(feature_df)
    store.save()
    registry.save()
    logger.info("Saved fake-factor features to %s.", feature_store_path)
    return feature_store_path


def calculate_and_store_classic_fake_factors(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    feature_store_path: Union[str, Path],
    feature_registry_path: Union[str, Path],
    fake_factors_path: Union[str, Path] = DEFAULT_PROCESS_FRACTIONS_PATH,
    corrections_path: Union[str, Path] = DEFAULT_CLASSIC_CORRECTIONS_PATH,
    regions: Iterable[str] = ("SR", "AR"),
) -> Path:
    """Calculate classic FFs and persist them as row-index keyed features."""
    feature_store_path = Path(feature_store_path)
    feature_registry_path = Path(feature_registry_path)
    fake_factors_path = Path(fake_factors_path)
    corrections_path = Path(corrections_path)
    regions = tuple(regions)

    for correction_path, description in (
        (fake_factors_path, "classic fake-factor"),
        (corrections_path, "classic FF correction"),
    ):
        if not correction_path.is_file():
            raise FileNotFoundError(
                f"Missing {description} file: {correction_path}"
            )

    df = load_data(data_path, masks_path)
    for region in regions:
        logger.info(
            "Classic FF application region %s contains %d events.",
            region,
            int(df.mask(region).sum()),
        )
    selected = _application_mask(df, regions)
    calculation_frame = df.events.loc[selected].copy()
    if calculation_frame.empty:
        raise ValueError(
            "No events selected for the classic fake-factor calculation."
        )

    calculate_fake_factor_classic(
        calculation_frame,
        fake_factors_path=fake_factors_path,
        corrections_path=corrections_path,
    )
    classic_values = calculation_frame["ff_classic"].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(classic_values).all():
        invalid_count = int((~np.isfinite(classic_values)).sum())
        raise ValueError(
            "Classic FF calculation produced "
            f"{invalid_count} non-finite values."
        )

    feature_df = pd.DataFrame({
        "row_index": calculation_frame.index,
        "event": calculation_frame["event"].to_numpy(),
        "ff_classic": classic_values,
    })
    registry = FeatureRegistry(feature_registry_path)
    store = FeatureStore(feature_store_path, registry)
    store.save()
    registry.save()
    logger.info(
        "Saved ff_classic for %d events to %s "
        "(minimum=%.8g, maximum=%.8g).",
        len(feature_df),
        feature_store_path,
        classic_values.min(),
        classic_values.max(),
    )
    return feature_store_path
