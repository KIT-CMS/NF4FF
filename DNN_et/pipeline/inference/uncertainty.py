"""Ensemble and dropout-mask fake-factor uncertainty inference."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch as t

from data.handling import (
    FeatureRegistry,
    FeatureStore,
    load_data,
    load_variables,
)
from models.networks import EnsembleStatUncWrapper, load_model
from core.paths import CONFIG_ROOT, PROJECT_ROOT
from inference.process import (
    EVALUATION_REGIONS,
    _application_mask,
    _model_features,
    _predict_fake_factors,
)


WORKFLOW_ROOT = PROJECT_ROOT / "Law_workflow_results"

DATA_PATH = WORKFLOW_ROOT / "data" / "dataframe_complete.feather"
MASKS_PATH = CONFIG_ROOT / "selections.yaml"
TRAINING_VARIABLES_PATH = CONFIG_ROOT / "variables_fake_factor.yaml"
COMBINED_MODELS_DIR = (
    WORKFLOW_ROOT / "CombinedModelsUncertainties" / "seeds_100_199"
)
FEATURE_REGISTRY_PATH = WORKFLOW_ROOT / "data" / "features" / "feature_registry.json"

GROUPING = "njets"
PROCESS = "wjets"
PROCESSES = ("wjets", "qcd", "ttbar")
PROCESS_MODEL_DIRS = {
    "wjets": "Wjets",
    "qcd": "QCD",
    "ttbar": "ttbar",
}
SEEDS = range(100, 200)
BATCH_SIZE = 10_000

# The requested order statistics, interpreted as 1-based positions.
FF_DOWN_INDEX = 15
FF_NOMINAL_INDEX = 49
FF_UP_INDEX = 83


def default_feature_store_path(process=PROCESS, feature_root=None):
    if feature_root is None:
        feature_root = WORKFLOW_ROOT / "data" / "features"
    return (
        Path(feature_root)
        / "fake_factor_model_uncertainty"
        / process
        / "fake_factor_model_uncertainty.feather"
    )


def default_dropout_mask_feature_store_path(feature_root=None):
    if feature_root is None:
        feature_root = WORKFLOW_ROOT / "data" / "features"
    return (
        Path(feature_root)
        / "fake_factor_dropout_mask_variation"
        / "fake_factor_dropout_mask_variation.feather"
    )


def output_columns(process=PROCESS, seeds=SEEDS):
    seeds = tuple(seeds)
    return (
        f"FF_{process}_down",
        f"FF_{process}_nominal",
        f"FF_{process}_up",
        *(f"FF_{process}_{index}" for index in range(len(seeds))),
    )


def dropout_mask_output_columns(n_masks=100):
    return (
        "FF_nominal_dmv",
        "FF_up_dmv",
        "FF_down_dmv",
        *(f"FF_{index}_dmv" for index in range(n_masks)),
    )


def model_path(
    seed,
    process=PROCESS,
    grouping=GROUPING,
    combined_models_dir=COMBINED_MODELS_DIR,
):
    return (
        Path(combined_models_dir)
        / PROCESS_MODEL_DIRS[process]
        / grouping
        / str(seed)
        / "torch_model"
    )


def load_uncertainty_models(
    process=PROCESS,
    grouping=GROUPING,
    seeds=SEEDS,
    combined_models_dir=COMBINED_MODELS_DIR,
    device="cpu",
):
    for seed_index, seed in enumerate(seeds):
        path = model_path(
            seed,
            process=process,
            grouping=grouping,
            combined_models_dir=combined_models_dir,
        )
        if not (path / "model_weights.pth").is_file():
            raise FileNotFoundError(f"Missing uncertainty combined model: {path}")
        yield seed_index, load_model(path, device=device)


def _set_dropout_mask(model, mask_index):
    found = 0
    for module in model.modules():
        if hasattr(module, "active_mask"):
            module.active_mask = mask_index
            found += 1
    if found == 0:
        raise ValueError("The selected model has no non-zero dropout layers.")


def calculate_ff_uncertainty_columns(
    process=PROCESS,
    seeds=SEEDS,
    combined_models_dir=COMBINED_MODELS_DIR,
):
    seeds = tuple(seeds)
    if len(seeds) <= FF_UP_INDEX:
        raise ValueError(
            "At least 84 models are required to calculate FF_up."
        )

    df = load_data(DATA_PATH, MASKS_PATH)
    training_variables = tuple(load_variables(TRAINING_VARIABLES_PATH))
    selected = _application_mask(df, EVALUATION_REGIONS)
    inference_frame = df.events.loc[selected].copy()
    if inference_frame.empty:
        raise ValueError("No events selected for fake-factor inference.")

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    predictions = np.full(
        (len(seeds), len(inference_frame)),
        np.nan,
        dtype=np.float32,
    )
    for model_index, model in load_uncertainty_models(
        process=process,
        seeds=seeds,
        combined_models_dir=combined_models_dir,
        device=str(device),
    ):
        model_features = _model_features(model, training_variables)
        prediction = _predict_fake_factors(
            model,
            inference_frame,
            model_features,
            batch_size=BATCH_SIZE,
            device=device,
        )
        predictions[model_index] = prediction
        column = f"FF_{process}_{model_index}"
        df.events[column] = np.nan
        df.events.loc[inference_frame.index, column] = prediction
        model.to("cpu")
        del model
        if device.type == "cuda":
            t.cuda.empty_cache()

    ordered = np.sort(predictions, axis=0)
    ff_down = ordered[FF_DOWN_INDEX]
    ff_nominal = ordered[FF_NOMINAL_INDEX]
    ff_up = ordered[FF_UP_INDEX]

    df.events[f"FF_{process}_down"] = np.nan
    df.events[f"FF_{process}_nominal"] = np.nan
    df.events[f"FF_{process}_up"] = np.nan
    df.events.loc[inference_frame.index, f"FF_{process}_down"] = ff_down
    df.events.loc[
        inference_frame.index,
        f"FF_{process}_nominal",
    ] = ff_nominal
    df.events.loc[inference_frame.index, f"FF_{process}_up"] = ff_up
    return df


def calculate_ff_dropout_mask_variation_columns(
    process=PROCESS,
    model_seed=100,
    n_masks=100,
    combined_models_dir=COMBINED_MODELS_DIR,
):
    if n_masks <= FF_UP_INDEX:
        raise ValueError(
            "At least 84 dropout masks are required to calculate FF_up_dmv."
        )

    df = load_data(DATA_PATH, MASKS_PATH)
    training_variables = tuple(load_variables(TRAINING_VARIABLES_PATH))
    selected = _application_mask(df, EVALUATION_REGIONS)
    inference_frame = df.events.loc[selected].copy()
    if inference_frame.empty:
        raise ValueError("No events selected for fake-factor inference.")

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    path = model_path(
        model_seed,
        process=process,
        combined_models_dir=combined_models_dir,
    )
    if not (path / "model_weights.pth").is_file():
        raise FileNotFoundError(
            f"Missing dropout-mask seed model: {path}"
        )

    t.manual_seed(model_seed)
    wrapper = EnsembleStatUncWrapper(
        model=load_model(path, device=str(device)),
        ensemble_size=n_masks,
        direction="Nominal",
    ).eval()
    masked_model = wrapper.wrapped_model
    model_features = _model_features(masked_model, training_variables)

    predictions = np.full(
        (n_masks, len(inference_frame)),
        np.nan,
        dtype=np.float32,
    )
    try:
        for mask_index in range(1, n_masks + 1):
            _set_dropout_mask(masked_model, mask_index)
            prediction = _predict_fake_factors(
                masked_model,
                inference_frame,
                model_features,
                batch_size=BATCH_SIZE,
                device=device,
            )
            output_index = mask_index - 1
            predictions[output_index] = prediction
            column = f"FF_{output_index}_dmv"
            df.events[column] = np.nan
            df.events.loc[inference_frame.index, column] = prediction
    finally:
        _set_dropout_mask(masked_model, None)
        masked_model.to("cpu")
        if device.type == "cuda":
            t.cuda.empty_cache()

    ordered = np.sort(predictions, axis=0)
    ff_down = ordered[FF_DOWN_INDEX]
    ff_nominal = ordered[FF_NOMINAL_INDEX]
    ff_up = ordered[FF_UP_INDEX]

    for column, values in (
        ("FF_down_dmv", ff_down),
        ("FF_nominal_dmv", ff_nominal),
        ("FF_up_dmv", ff_up),
    ):
        df.events[column] = np.nan
        df.events.loc[inference_frame.index, column] = values
    return df


def calculate_ff_uncertainty_features(
    process=PROCESS,
    seeds=SEEDS,
    combined_models_dir=COMBINED_MODELS_DIR,
):
    df = calculate_ff_uncertainty_columns(
        process=process,
        seeds=seeds,
        combined_models_dir=combined_models_dir,
    )
    feature_df = pd.DataFrame({
        "row_index": df.events.index,
        "event": df.events["event"].to_numpy(),
    })
    for column in output_columns(process=process, seeds=seeds):
        feature_df[column] = df.events[column].to_numpy()
    return feature_df


def calculate_ff_dropout_mask_variation_features(
    process=PROCESS,
    model_seed=100,
    n_masks=100,
    combined_models_dir=COMBINED_MODELS_DIR,
):
    df = calculate_ff_dropout_mask_variation_columns(
        process=process,
        model_seed=model_seed,
        n_masks=n_masks,
        combined_models_dir=combined_models_dir,
    )
    feature_df = pd.DataFrame({
        "row_index": df.events.index,
        "event": df.events["event"].to_numpy(),
    })
    for column in dropout_mask_output_columns(n_masks=n_masks):
        feature_df[column] = df.events[column].to_numpy()
    return feature_df


def _check_feature_output_is_free(
    feature_store_path=None,
    feature_registry_path=FEATURE_REGISTRY_PATH,
    columns=None,
    overwrite=False,
):
    if feature_store_path is None:
        feature_store_path = default_feature_store_path()
    feature_store_path = Path(feature_store_path)
    feature_registry_path = Path(feature_registry_path)
    if columns is None:
        columns = output_columns()
    if feature_store_path.exists() and not overwrite:
        raise FileExistsError(
            "Feature output already exists. Refusing to overwrite: "
            f"{feature_store_path}"
        )

    registry = FeatureRegistry(feature_registry_path)
    conflicts = {
        column: registered_path
        for column in columns
        if (
            registered_path := registry.get_file(column)
        ) is not None
        and Path(registered_path).resolve() != feature_store_path.resolve()
    }
    if conflicts:
        formatted = "\n".join(
            f"  {column}: {path}"
            for column, path in sorted(conflicts.items())
        )
        raise ValueError(
            "Refusing to overwrite existing feature registry entries:\n"
            f"{formatted}"
        )
    return registry


def calculate_and_store_ff_uncertainty_features(
    process=PROCESS,
    feature_store_path=None,
    feature_registry_path=FEATURE_REGISTRY_PATH,
    seeds=SEEDS,
    combined_models_dir=COMBINED_MODELS_DIR,
    overwrite=False,
):
    seeds = tuple(seeds)
    if feature_store_path is None:
        feature_store_path = default_feature_store_path(process)
    registry = _check_feature_output_is_free(
        feature_store_path=feature_store_path,
        feature_registry_path=feature_registry_path,
        columns=output_columns(process=process, seeds=seeds),
        overwrite=overwrite,
    )
    feature_df = calculate_ff_uncertainty_features(
        process=process,
        seeds=seeds,
        combined_models_dir=combined_models_dir,
    )
    store = FeatureStore(feature_store_path, registry)
    store.write(feature_df)
    store.save()
    registry.save()
    return Path(feature_store_path)

def calculate_and_store_ff_dropout_mask_variation_features(
    process=PROCESS,
    feature_store_path=None,
    feature_registry_path=FEATURE_REGISTRY_PATH,
    model_seed=100,
    n_masks=100,
    combined_models_dir=COMBINED_MODELS_DIR,
    overwrite=False,
):
    if feature_store_path is None:
        feature_store_path = default_dropout_mask_feature_store_path()
    registry = _check_feature_output_is_free(
        feature_store_path=feature_store_path,
        feature_registry_path=feature_registry_path,
        columns=dropout_mask_output_columns(n_masks=n_masks),
        overwrite=overwrite,
    )
    feature_df = calculate_ff_dropout_mask_variation_features(
        process=process,
        model_seed=model_seed,
        n_masks=n_masks,
        combined_models_dir=combined_models_dir,
    )
    store = FeatureStore(feature_store_path, registry)
    store.write(feature_df)
    store.save()
    registry.save()
    return Path(feature_store_path)
