import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch as t

from classes import (
    EnsembleStatUncWrapper,
    create_training_dataset,
    load_data,
    load_model,
    load_variables,
)
from classes.DataHandling import FeatureRegistry, FeatureStore
from ff_calculation import EVALUATION_REGIONS, _application_mask, _model_features
from ff_model_uncertainty import (
    COMBINED_MODELS_DIR,
    DATA_PATH,
    FEATURE_REGISTRY_PATH,
    GROUPING,
    MASKS_PATH,
    PROCESS,
    SEEDS,
    TRAINING_VARIABLES_PATH,
    load_uncertainty_models,
    model_path,
)
from groupings import grouping_bounds, grouping_source
from training_squeezed_loss import (
    REDUCED_WEIGHT_DIR,
    SEED,
    _prepare_training_frame,
)


logger = logging.getLogger(__name__)

BATCH_SIZE = 2048
EPSILON = 0.0


def default_gradient_covariance_feature_store_path(
    process=PROCESS,
    feature_root=None,
    seeds=SEEDS,
):
    if feature_root is None:
        feature_root = Path(FEATURE_REGISTRY_PATH).parent
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("At least one seed is required.")
    return (
        Path(feature_root)
        / "fake_factor_gradient_covariance_uncertainty"
        / process
        / f"seeds_{seeds[0]}_{seeds[-1]}"
        / "fake_factor_gradient_covariance_uncertainty.feather"
    )


def default_gradient_covariance_dropout_mask_feature_store_path(
    process=PROCESS,
    feature_root=None,
    model_seed=100,
    n_masks=100,
):
    if feature_root is None:
        feature_root = Path(FEATURE_REGISTRY_PATH).parent
    return (
        Path(feature_root)
        / "fake_factor_gradient_covariance_dropout_mask_variation"
        / process
        / f"seed_{model_seed}"
        / f"n_masks_{n_masks}"
        / "fake_factor_gradient_covariance_dropout_mask_variation.feather"
    )


def gradient_covariance_output_columns(process=PROCESS):
    return (
        f"FF_{process}_gradcov_sigma_sum",
        f"FF_{process}_gradcov_sigma_mean",
        f"FF_{process}_gradcov_variance_sum",
        f"FF_{process}_gradcov_variance_mean",
    )


def gradient_covariance_dropout_mask_output_columns(process=PROCESS):
    return (
        f"FF_{process}_gradcov_sigma_sum_dmv",
        f"FF_{process}_gradcov_sigma_mean_dmv",
        f"FF_{process}_gradcov_variance_sum_dmv",
        f"FF_{process}_gradcov_variance_mean_dmv",
    )


def _wjets_njets_training_frames(
    *,
    data_path,
    masks_path,
    training_variables_path,
    reduced_weight_dir,
):
    training_variables = list(load_variables(training_variables_path))
    grouping_name = "njets"
    process = "wjets"
    source_weight = f"reduced_weight_{process}_{grouping_name}_nominal"
    weight_column = "weight_wjets"

    data = load_data(data_path, masks_path)
    data.load_feature_file(
        Path(reduced_weight_dir)
        / process
        / f"reduced_weight_{grouping_name}.feather"
    )
    signal = _prepare_training_frame(
        data.data.SR_like_wjets.events,
        training_variables,
        source_weight,
        weight_column,
        "njets/wjets/SR-like",
    )
    background = _prepare_training_frame(
        data.data.AR_like_wjets.events,
        training_variables,
        source_weight,
        weight_column,
        "njets/wjets/AR-like",
    )
    return signal, background, training_variables, weight_column


def _fold_training_dataset_features(
    signal,
    background,
    training_variables,
    weight_column,
    *,
    applied_fold,
):
    if applied_fold == "fold_even":
        train_signal = signal[signal["event"] % 2 == 1]
        train_background = background[background["event"] % 2 == 1]
    elif applied_fold == "fold_odd":
        train_signal = signal[signal["event"] % 2 == 0]
        train_background = background[background["event"] % 2 == 0]
    else:
        raise ValueError(f"Unsupported fold: {applied_fold}")

    train, _ = create_training_dataset(
        df_sig=train_signal,
        df_bkg=train_background,
        training_var=training_variables,
        weight_column=weight_column,
        balance=True,
        balance_column=grouping_source("njets"),
        balance_groups=grouping_bounds("njets", "wjets"),
        balance_with_absolute_yields=True,
        test_size=0.25,
        random_state=SEED,
    )
    features = train.X.detach().cpu().numpy().astype(np.float64, copy=False)
    weights = (
        train.weights.detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
        .reshape(-1)
    )
    return features, weights


def _normalized_covariance_matrix(
    values: np.ndarray,
    *,
    weights: np.ndarray,
    epsilon: float = EPSILON,
):
    if values.ndim != 2:
        raise ValueError(f"Expected 2D feature values, got shape {values.shape}")
    if len(values) < 2:
        raise ValueError("At least two training rows are needed for covariance")
    if weights.ndim != 1 or len(weights) != len(values):
        raise ValueError(
            "Covariance weights must be one-dimensional and match the "
            f"number of rows: weights={weights.shape}, values={values.shape}"
        )
    if not np.isfinite(weights).all():
        raise ValueError("Covariance weights contain non-finite values.")
    if (weights < 0).any():
        raise ValueError("Covariance weights must be non-negative.")
    if weights.sum(dtype=np.float64) <= 0:
        raise ValueError("Covariance weights must have a positive sum.")
    covariance = np.cov(values, rowvar=False, aweights=weights)
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    if epsilon:
        covariance = covariance + epsilon * np.eye(covariance.shape[0])
    scale = np.sqrt(np.diag(covariance))
    if (scale <= 0).any() or not np.isfinite(scale).all():
        raise ValueError(
            "Cannot normalize covariance with non-positive or non-finite "
            "feature scales."
        )
    normalized_covariance = covariance / np.outer(scale, scale)
    return (
        normalized_covariance.astype(np.float32, copy=False),
        scale.astype(np.float32, copy=False),
    )


def _write_covariance_matrices(
    covariance_payload: Dict[str, Dict[str, np.ndarray]],
    training_variables: Tuple[str, ...],
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "coordinate_system": (
            "z_i = (x_i - weighted_mean_i) / weighted_covariance_std_i"
        ),
        "variables": list(training_variables),
        "folds": {},
    }
    for fold, payload in covariance_payload.items():
        covariance = payload["covariance"]
        scale = payload["scale"]
        covariance_path = output_dir / f"{fold}_normalized_covariance.csv"
        scale_path = output_dir / f"{fold}_normalization_scale.csv"
        pd.DataFrame(
            covariance,
            index=training_variables,
            columns=training_variables,
        ).to_csv(covariance_path)
        pd.DataFrame({
            "variable": training_variables,
            "scale": scale,
        }).to_csv(scale_path, index=False)
        metadata["folds"][fold] = {
            "normalized_covariance_csv": covariance_path.name,
            "normalization_scale_csv": scale_path.name,
        }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output_dir


def wjets_njets_training_covariances(
    *,
    data_path=DATA_PATH,
    masks_path=MASKS_PATH,
    training_variables_path=TRAINING_VARIABLES_PATH,
    reduced_weight_dir=REDUCED_WEIGHT_DIR,
    epsilon: float = EPSILON,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Tuple[str, ...]]:
    signal, background, training_variables, weight_column = (
        _wjets_njets_training_frames(
            data_path=data_path,
            masks_path=masks_path,
            training_variables_path=training_variables_path,
            reduced_weight_dir=reduced_weight_dir,
        )
    )
    covariances = {}
    for fold in ("fold_even", "fold_odd"):
        features, weights = _fold_training_dataset_features(
            signal,
            background,
            training_variables,
            weight_column,
            applied_fold=fold,
        )
        covariance, scale = _normalized_covariance_matrix(
            features,
            weights=weights,
            epsilon=epsilon,
        )
        covariances[fold] = {
            "covariance": covariance,
            "scale": scale,
        }
        logger.info(
            "%s weighted normalized covariance calculated from %d rows, "
            "%d variables, and weight sum %.6g",
            fold,
            features.shape[0],
            features.shape[1],
            weights.sum(dtype=np.float64),
        )
    return covariances, tuple(training_variables)


def _gradient_sum_for_model(
    model,
    frame: pd.DataFrame,
    feature_names: Tuple[str, ...],
    *,
    batch_size: int,
    device: t.device,
) -> np.ndarray:
    feature_values = frame.loc[:, list(feature_names)].to_numpy(dtype=np.float32)
    event_values = frame["event"].to_numpy(dtype=np.float32)
    finite = np.isfinite(feature_values).all(axis=1) & np.isfinite(event_values)
    gradients = np.zeros(
        (len(frame), len(feature_names)),
        dtype=np.float32,
    )
    valid_indices = np.flatnonzero(finite)
    if len(valid_indices) == 0:
        raise ValueError("No finite events are available for gradients.")

    model = model.eval().to(device)
    for start in range(0, len(valid_indices), batch_size):
        batch_indices = valid_indices[start:start + batch_size]
        features = t.as_tensor(
            feature_values[batch_indices],
            dtype=t.float32,
            device=device,
        ).requires_grad_(True)
        event_parity = t.as_tensor(
            event_values[batch_indices] % 2,
            dtype=t.float32,
            device=device,
        )
        model_input = t.cat([event_parity.unsqueeze(0), features.T], dim=0)
        output = model(model_input).reshape(-1)
        if len(output) != len(batch_indices):
            raise RuntimeError(
                "FF model returned an unexpected number of predictions: "
                f"{len(output)} for {len(batch_indices)} events."
            )
        batch_gradient = t.autograd.grad(output.sum(), features)[0]
        gradients[batch_indices] = batch_gradient.detach().cpu().numpy()

    return gradients


def _fold_quadratic_form(
    gradients: np.ndarray,
    covariances: Dict[str, Dict[str, np.ndarray]],
    events: np.ndarray,
) -> np.ndarray:
    variance = np.full(len(gradients), np.nan, dtype=np.float32)
    for fold, parity in (("fold_even", 0), ("fold_odd", 1)):
        mask = (events.astype(np.int64) % 2) == parity
        if not mask.any():
            continue
        covariance = covariances[fold]["covariance"]
        scale = covariances[fold]["scale"]
        normalized_gradients = gradients[mask] * scale
        values = np.einsum(
            "bi,ij,bj->b",
            normalized_gradients,
            covariance,
            normalized_gradients,
            optimize=True,
        )
        variance[mask] = np.maximum(values, 0.0)
    return variance


def _set_dropout_mask(model, mask_index):
    found = 0
    for module in model.modules():
        if hasattr(module, "active_mask"):
            module.active_mask = mask_index
            found += 1
    if found == 0:
        raise ValueError("The selected model has no non-zero dropout layers.")


def calculate_ff_gradient_covariance_features(
    *,
    process=PROCESS,
    grouping=GROUPING,
    seeds=SEEDS,
    data_path=DATA_PATH,
    masks_path=MASKS_PATH,
    training_variables_path=TRAINING_VARIABLES_PATH,
    reduced_weight_dir=REDUCED_WEIGHT_DIR,
    combined_models_dir=COMBINED_MODELS_DIR,
    regions: Iterable[str] = EVALUATION_REGIONS,
    batch_size: int = BATCH_SIZE,
    epsilon: float = EPSILON,
    covariance_output_dir=None,
):
    if process != "wjets" or grouping != "njets":
        raise ValueError(
            "Gradient-covariance uncertainty is currently implemented for "
            "process='wjets' and grouping='njets'."
        )
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("At least one seed is required.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    covariances, training_variables = wjets_njets_training_covariances(
        data_path=data_path,
        masks_path=masks_path,
        training_variables_path=training_variables_path,
        reduced_weight_dir=reduced_weight_dir,
        epsilon=epsilon,
    )
    if covariance_output_dir is not None:
        _write_covariance_matrices(
            covariances,
            training_variables,
            covariance_output_dir,
        )
    data = load_data(data_path, masks_path)
    selected = _application_mask(data, regions)
    frame = data.events.loc[selected].copy()
    if frame.empty:
        raise ValueError("No events selected for FF gradient calculation.")
    finite_inputs = (
        np.isfinite(
            frame.loc[:, list(training_variables)].to_numpy(dtype=np.float32)
        ).all(axis=1)
        & np.isfinite(frame["event"].to_numpy(dtype=np.float32))
    )

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    gradient_sum = np.zeros(
        (len(frame), len(training_variables)),
        dtype=np.float32,
    )
    started_at = time.monotonic()
    for seed_index, model in load_uncertainty_models(
        process=process,
        grouping=grouping,
        seeds=seeds,
        combined_models_dir=combined_models_dir,
        device=str(device),
    ):
        model_features = _model_features(model, training_variables)
        gradient_sum += _gradient_sum_for_model(
            model,
            frame,
            model_features,
            batch_size=batch_size,
            device=device,
        )
        logger.info(
            "%s/%s seed %d/%d gradients accumulated",
            process,
            grouping,
            seed_index + 1,
            len(seeds),
        )
        model.to("cpu")
        del model
        if device.type == "cuda":
            t.cuda.empty_cache()

    variance_sum = _fold_quadratic_form(
        gradient_sum,
        covariances,
        frame["event"].to_numpy(),
    )
    variance_sum[~finite_inputs] = np.nan
    variance_mean = variance_sum / (len(seeds) ** 2)
    sigma_sum = np.sqrt(variance_sum).astype(np.float32, copy=False)
    sigma_mean = np.sqrt(variance_mean).astype(np.float32, copy=False)

    logger.info(
        "Gradient-covariance uncertainty finished in %.1f min",
        (time.monotonic() - started_at) / 60.0,
    )
    return pd.DataFrame({
        "row_index": frame.index,
        "event": frame["event"].to_numpy(),
        f"FF_{process}_gradcov_sigma_sum": sigma_sum,
        f"FF_{process}_gradcov_sigma_mean": sigma_mean,
        f"FF_{process}_gradcov_variance_sum": variance_sum,
        f"FF_{process}_gradcov_variance_mean": variance_mean,
    })


def calculate_ff_gradient_covariance_dropout_mask_variation_features(
    *,
    process=PROCESS,
    grouping=GROUPING,
    model_seed=100,
    n_masks=100,
    data_path=DATA_PATH,
    masks_path=MASKS_PATH,
    training_variables_path=TRAINING_VARIABLES_PATH,
    reduced_weight_dir=REDUCED_WEIGHT_DIR,
    combined_models_dir=COMBINED_MODELS_DIR,
    regions: Iterable[str] = EVALUATION_REGIONS,
    batch_size: int = BATCH_SIZE,
    epsilon: float = EPSILON,
    covariance_output_dir=None,
):
    if process != "wjets" or grouping != "njets":
        raise ValueError(
            "Gradient-covariance dropout-mask variation is currently "
            "implemented for process='wjets' and grouping='njets'."
        )
    if n_masks <= 0:
        raise ValueError(f"n_masks must be positive, got {n_masks}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    covariances, training_variables = wjets_njets_training_covariances(
        data_path=data_path,
        masks_path=masks_path,
        training_variables_path=training_variables_path,
        reduced_weight_dir=reduced_weight_dir,
        epsilon=epsilon,
    )
    if covariance_output_dir is not None:
        _write_covariance_matrices(
            covariances,
            training_variables,
            covariance_output_dir,
        )
    data = load_data(data_path, masks_path)
    selected = _application_mask(data, regions)
    frame = data.events.loc[selected].copy()
    if frame.empty:
        raise ValueError("No events selected for FF gradient calculation.")
    finite_inputs = (
        np.isfinite(
            frame.loc[:, list(training_variables)].to_numpy(dtype=np.float32)
        ).all(axis=1)
        & np.isfinite(frame["event"].to_numpy(dtype=np.float32))
    )

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    path = model_path(
        model_seed,
        process=process,
        grouping=grouping,
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
    masked_model = wrapper.wrapped_model.to(device)
    model_features = _model_features(masked_model, training_variables)
    gradient_sum = np.zeros(
        (len(frame), len(training_variables)),
        dtype=np.float32,
    )

    started_at = time.monotonic()
    try:
        for mask_index in range(1, n_masks + 1):
            _set_dropout_mask(masked_model, mask_index)
            gradient_sum += _gradient_sum_for_model(
                masked_model,
                frame,
                model_features,
                batch_size=batch_size,
                device=device,
            )
            logger.info(
                "%s/%s dropout mask %d/%d gradients accumulated",
                process,
                grouping,
                mask_index,
                n_masks,
            )
    finally:
        _set_dropout_mask(masked_model, None)
        masked_model.to("cpu")
        if device.type == "cuda":
            t.cuda.empty_cache()

    variance_sum = _fold_quadratic_form(
        gradient_sum,
        covariances,
        frame["event"].to_numpy(),
    )
    variance_sum[~finite_inputs] = np.nan
    variance_mean = variance_sum / (n_masks ** 2)
    sigma_sum = np.sqrt(variance_sum).astype(np.float32, copy=False)
    sigma_mean = np.sqrt(variance_mean).astype(np.float32, copy=False)

    logger.info(
        "Gradient-covariance dropout-mask variation finished in %.1f min",
        (time.monotonic() - started_at) / 60.0,
    )
    return pd.DataFrame({
        "row_index": frame.index,
        "event": frame["event"].to_numpy(),
        f"FF_{process}_gradcov_sigma_sum_dmv": sigma_sum,
        f"FF_{process}_gradcov_sigma_mean_dmv": sigma_mean,
        f"FF_{process}_gradcov_variance_sum_dmv": variance_sum,
        f"FF_{process}_gradcov_variance_mean_dmv": variance_mean,
    })


def _check_feature_output_is_free(
    feature_store_path,
    feature_registry_path,
    *,
    columns,
    overwrite=False,
):
    feature_store_path = Path(feature_store_path)
    feature_registry_path = Path(feature_registry_path)
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


def calculate_and_store_ff_gradient_covariance_features(
    *,
    process=PROCESS,
    grouping=GROUPING,
    feature_store_path=None,
    feature_registry_path=FEATURE_REGISTRY_PATH,
    seeds=SEEDS,
    covariance_output_dir=None,
    overwrite=False,
    **kwargs,
):
    if feature_store_path is None:
        feature_store_path = default_gradient_covariance_feature_store_path(
            process=process,
            seeds=seeds,
        )
    if covariance_output_dir is None:
        covariance_output_dir = Path(feature_store_path).parent / "covariances"
    registry = _check_feature_output_is_free(
        feature_store_path,
        feature_registry_path,
        columns=gradient_covariance_output_columns(process),
        overwrite=overwrite,
    )
    feature_df = calculate_ff_gradient_covariance_features(
        process=process,
        grouping=grouping,
        seeds=seeds,
        covariance_output_dir=covariance_output_dir,
        **kwargs,
    )
    store = FeatureStore(feature_store_path, registry)
    store.write(feature_df)
    store.save()
    registry.save()
    return Path(feature_store_path)


def calculate_and_store_ff_gradient_covariance_dropout_mask_variation_features(
    *,
    process=PROCESS,
    grouping=GROUPING,
    feature_store_path=None,
    feature_registry_path=FEATURE_REGISTRY_PATH,
    model_seed=100,
    n_masks=100,
    covariance_output_dir=None,
    overwrite=False,
    **kwargs,
):
    if feature_store_path is None:
        feature_store_path = (
            default_gradient_covariance_dropout_mask_feature_store_path(
                process=process,
                model_seed=model_seed,
                n_masks=n_masks,
            )
        )
    if covariance_output_dir is None:
        covariance_output_dir = Path(feature_store_path).parent / "covariances"
    registry = _check_feature_output_is_free(
        feature_store_path,
        feature_registry_path,
        columns=gradient_covariance_dropout_mask_output_columns(process),
        overwrite=overwrite,
    )
    feature_df = (
        calculate_ff_gradient_covariance_dropout_mask_variation_features(
            process=process,
            grouping=grouping,
            model_seed=model_seed,
            n_masks=n_masks,
            covariance_output_dir=covariance_output_dir,
            **kwargs,
        )
    )
    store = FeatureStore(feature_store_path, registry)
    store.write(feature_df)
    store.save()
    registry.save()
    return Path(feature_store_path)


if __name__ == "__main__":
    calculate_and_store_ff_gradient_covariance_features()
