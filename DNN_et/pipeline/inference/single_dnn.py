"""Single-DNN model export and fake-factor inference."""

import json
import logging
from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np
import pandas as pd
import torch as t

from models.networks import (
    FoldCombinedDNN,
    LikelihoodRatioCalculation,
    convert_models_to_onnx,
    load_model,
    save_model,
)
from data.handling import (
    FeatureRegistry,
    FeatureStore,
    load_data,
    load_variables,
)
from inference.process import (
    DEFAULT_PROCESS_FRACTIONS_PATH,
    EVALUATION_REGIONS,
    _application_mask,
    _model_features,
    _predict_fake_factors,
)


logger = logging.getLogger(__name__)
PROCESSES = ("wjets", "qcd", "ttbar")
PROCESS_OUTPUT_NAMES = {
    "wjets": "Wjets",
    "qcd": "QCD",
    "ttbar": "ttbar",
}
FOLD_PARITIES = {"fold_even": 0, "fold_odd": 1}


def _process_frames(df, process):
    if process == "ttbar":
        return (
            df.ttbar.SR_like_ttbar.events,
            df.ttbar.AR_like_ttbar.events,
            "weight",
        )
    return (
        getattr(df.data, f"SR_like_{process}").events,
        getattr(df.data, f"AR_like_{process}").events,
        f"weight_{process}",
    )


def _inclusive_normalization(df, process, parity=None):
    signal, application, weight_column = _process_frames(df, process)
    signal_mask = np.ones(len(signal), dtype=bool)
    application_mask = np.ones(len(application), dtype=bool)
    if parity is not None:
        signal_mask &= signal["event"].to_numpy() % 2 == parity
        application_mask &= application["event"].to_numpy() % 2 == parity
    numerator = float(signal.loc[signal_mask, weight_column].sum())
    denominator = float(
        application.loc[application_mask, weight_column].sum()
    )
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError(
            f"Cannot normalize single DNN for {process}: "
            f"AR-like yield is {denominator}."
        )
    return numerator / denominator, {
        "sr_yield": numerator,
        "ar_yield": denominator,
        "sr_events": int(signal_mask.sum()),
        "ar_events": int(application_mask.sum()),
    }


def convert_single_dnn_models(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    trained_models_dir: Union[str, Path],
    reduced_weight_dir: Union[str, Path],
    reduced_weight_grouping: str,
    output_dir: Union[str, Path],
):
    trained_models_dir = Path(trained_models_dir)
    reduced_weight_dir = Path(reduced_weight_dir)
    output_dir = Path(output_dir)
    df = load_data(data_path, masks_path)
    for process in ("wjets", "qcd"):
        df.load_feature_file(
            reduced_weight_dir
            / process
            / f"reduced_weight_{reduced_weight_grouping}.feather"
        )
        source = (
            f"reduced_weight_{process}_"
            f"{reduced_weight_grouping}_nominal"
        )
        df.events[f"weight_{process}"] = df.events[source]

    constants = {}
    diagnostics = {}
    outputs = []
    for process in PROCESSES:
        constants[process] = {}
        diagnostics[process] = {}
        fold_models = {}
        for fold, parity in FOLD_PARITIES.items():
            value, fold_diagnostics = _inclusive_normalization(
                df,
                process,
                parity,
            )
            constants[process][fold] = value
            diagnostics[process][fold] = fold_diagnostics
            model_path = trained_models_dir / process / fold
            fold_models[fold] = LikelihoodRatioCalculation(
                model=load_model(model_path).eval(),
                normalization_constants=value,
                clip=(1e-4, 10.0),
            )

        combined = FoldCombinedDNN(
            even_model=fold_models["fold_even"],
            odd_model=fold_models["fold_odd"],
            fold_id_name="event",
        )
        process_dir = output_dir / PROCESS_OUTPUT_NAMES[process]
        save_model(combined, process_dir / "torch_model")
        onnx_path = process_dir / "onnx_model" / "model.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        convert_models_to_onnx(
            torch_model=combined,
            onnx_model_path=onnx_path,
        )
        outputs.append(onnx_path)

    normalization_path = output_dir / "normalization_constants.json"
    normalization_path.parent.mkdir(parents=True, exist_ok=True)
    normalization_path.write_text(json.dumps({
        "constants": constants,
        "diagnostics": diagnostics,
        "grouping": None,
        "reduced_weight_grouping": reduced_weight_grouping,
    }, indent=2) + "\n")
    return [normalization_path, *outputs]


def calculate_single_dnn_fake_factors(
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
):
    combined_models_dir = Path(combined_models_dir)
    df = load_data(data_path, masks_path)
    selected = _application_mask(df, tuple(regions))
    inference_frame = df.events.loc[selected].copy()
    training_variables = tuple(load_variables(training_variables_path))
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    feature_df = pd.DataFrame({
        "row_index": inference_frame.index,
        "event": inference_frame["event"].to_numpy(),
    })

    for process in PROCESSES:
        model_path = (
            combined_models_dir
            / PROCESS_OUTPUT_NAMES[process]
            / "torch_model"
        )
        model = load_model(model_path, device=str(device))
        model_features = _model_features(model, training_variables)
        name = f"ff_dnn_single_{process}"
        feature_df[name] = _predict_fake_factors(
            model,
            inference_frame,
            model_features,
            batch_size=batch_size,
            device=device,
        )

    try:
        import correctionlib as cr
    except ImportError as error:
        raise ImportError(
            "Single-DNN FF calculation requires correctionlib."
        ) from error
    fractions = cr.CorrectionSet.from_file(
        str(process_fractions_path)
    )["process_fractions"]

    sr_ar_mask = df.mask("SR") | df.mask("AR")
    sr_ar_rows = inference_frame.index.intersection(
        df.events.index[sr_ar_mask]
    )
    sr_ar = inference_frame.loc[sr_ar_rows].copy()
    values = feature_df.set_index("row_index")
    for process in PROCESSES:
        sr_ar[f"ff_dnn_single_{process}"] = values.loc[
            sr_ar.index,
            f"ff_dnn_single_{process}",
        ]
    process_fractions = {
        "wjets": fractions.evaluate(
            "Wjets", sr_ar["mt_1"], sr_ar["njets"], "nominal"
        ),
        "qcd": fractions.evaluate(
            "QCD", sr_ar["mt_1"], sr_ar["njets"], "nominal"
        ),
        "ttbar": fractions.evaluate(
            "ttbar", sr_ar["mt_1"], sr_ar["njets"], "nominal"
        ),
    }
    combined = sum(
        process_fractions[process]
        * sr_ar[f"ff_dnn_single_{process}"].to_numpy()
        for process in PROCESSES
    )
    feature_df["ff_dnn_single"] = feature_df["row_index"].map(
        pd.Series(combined, index=sr_ar.index)
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
    return Path(feature_store_path)
