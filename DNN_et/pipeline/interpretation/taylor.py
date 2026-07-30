"""Taylor-coefficient analysis and plotting."""

import json
import logging
import math
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as t

from models.networks import (
    DNN,
    FoldCombinedDNN,
    GroupedDNN,
    load_fold_combined_model,
    load_model,
    temporary_extract_scaler_callable,
)
from data.handling import load_data, load_variables

logger = logging.getLogger(__name__)

Model = Union[DNN, GroupedDNN, FoldCombinedDNN, t.nn.Module]
CoefficientDict = Dict[str, Dict[str, float]]

TAYLOR_CATEGORIES = (
    ("inclusive", None),
    ("njets_eq_0", ("njets", "eq", 0)),
    ("njets_eq_1", ("njets", "eq", 1)),
    ("njets_ge_2", ("njets", "ge", 2)),
    ("tau_decaymode_2_eq_0", ("tau_decaymode_2", "eq", 0)),
    ("tau_decaymode_2_eq_1", ("tau_decaymode_2", "eq", 1)),
    ("tau_decaymode_2_eq_10", ("tau_decaymode_2", "eq", 10)),
    ("tau_decaymode_2_eq_11", ("tau_decaymode_2", "eq", 11)),
    ("tau_decaymode_2_in_0_1", ("tau_decaymode_2", "isin", (0, 1))),
    ("tau_decaymode_2_in_10_11", ("tau_decaymode_2", "isin", (10, 11))),
)
TAYLOR_CATEGORY_SELECTIONS = dict(TAYLOR_CATEGORIES)
TAYLOR_GROUPING_COLORS = {
    "tau_decaymode_2": "#1b9e77",
    "tau_decaymode_2_alt": "#d95f02",
    "njets": "#7570b3",
}
DEFAULT_TAYLOR_COLOR = "#4c78a8"


def _legacy_coefficient_dict(
    coefficients: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    """Return the flat JSON format consumed by the existing viewer tool."""
    return {
        **coefficients["first_order"],
        **coefficients["second_order"],
    }


def _model_device(model: t.nn.Module) -> t.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else t.device("cpu")


def _feature_indices(model: t.nn.Module, n_features: int) -> Tuple[int, ...]:
    offset = 1 if isinstance(model, FoldCombinedDNN) else 0
    return tuple(range(offset, offset + n_features))


def _model_input(
    model: t.nn.Module,
    features: t.Tensor,
    event_ids: Optional[t.Tensor],
) -> t.Tensor:
    if isinstance(model, FoldCombinedDNN):
        if event_ids is None:
            raise ValueError("event_ids are required for a FoldCombinedDNN")
        return t.cat((event_ids.reshape(1, -1), features.T), dim=0)
    return features


def _mean_absolute_taylor_sums(
    model: t.nn.Module,
    model_input: t.Tensor,
    coefficient_indices: Sequence[int],
    max_order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model_input = model_input.detach().requires_grad_(True)
    output = model(model_input)
    if output.numel() == 0:
        raise ValueError("Cannot calculate Taylor coefficients for an empty batch")

    first_gradient = t.autograd.grad(
        output.sum(),
        model_input,
        create_graph=max_order >= 2,
    )[0]
    first = np.asarray([
        first_gradient[index].abs().sum().detach().cpu().item()
        if isinstance(model, FoldCombinedDNN)
        else first_gradient[:, index].abs().sum().detach().cpu().item()
        for index in coefficient_indices
    ])

    second = np.zeros((len(coefficient_indices), len(coefficient_indices)))
    if max_order >= 2:
        for local_i, input_i in enumerate(coefficient_indices):
            gradient_i = (
                first_gradient[input_i]
                if isinstance(model, FoldCombinedDNN)
                else first_gradient[:, input_i]
            )
            second_gradient = t.autograd.grad(
                gradient_i.sum(),
                model_input,
                retain_graph=local_i < len(coefficient_indices) - 1,
            )[0]
            for local_j, input_j in enumerate(
                coefficient_indices[local_i:],
                start=local_i,
            ):
                values = (
                    second_gradient[input_j]
                    if isinstance(model, FoldCombinedDNN)
                    else second_gradient[:, input_j]
                )
                factorial = 2.0 if local_i == local_j else 1.0
                second[local_i, local_j] = (
                    values.abs().sum().detach().cpu().item() / factorial
                )

    return first, second


def calculate_taylor_coefficients(
    model: Model,
    features: Union[np.ndarray, t.Tensor],
    feature_names: Sequence[str],
    *,
    event_ids: Optional[Union[np.ndarray, t.Tensor]] = None,
    max_order: int = 2,
    batch_size: int = 1024,
    progress_label: Optional[str] = None,
    progress_interval_seconds: float = 60.0,
) -> CoefficientDict:
    """Calculate mean absolute first- and second-order Taylor coefficients.

    The returned second-order diagonal terms include the Taylor factorial
    factor ``1 / 2!``. For fold-combined models, ``event_ids`` are used only
    for fold routing and are not included in the coefficient output.
    """
    if max_order not in (1, 2):
        raise ValueError(f"max_order must be 1 or 2, got {max_order}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    features = t.as_tensor(features, dtype=t.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be two-dimensional, got {features.shape}")
    if features.shape[1] != len(feature_names):
        raise ValueError(
            f"Received {features.shape[1]} features but {len(feature_names)} names"
        )

    if event_ids is not None:
        event_ids = t.as_tensor(event_ids, dtype=features.dtype).reshape(-1)
        if len(event_ids) != len(features):
            raise ValueError("event_ids and features must have the same length")

    device = _model_device(model)
    total_batches = math.ceil(len(features) / batch_size)
    started_at = time.monotonic()
    last_progress_at = started_at
    if progress_label:
        logger.info(
            "%s: starting Taylor calculation for %d events in %d batches "
            "(%d features, order %d, device=%s, CPU threads=%d)",
            progress_label,
            len(features),
            total_batches,
            len(feature_names),
            max_order,
            device,
            t.get_num_threads(),
        )
    coefficient_indices = _feature_indices(model, len(feature_names))
    first_sum = np.zeros(len(feature_names), dtype=np.float64)
    second_sum = np.zeros(
        (len(feature_names), len(feature_names)),
        dtype=np.float64,
    )

    model.eval()
    with _temporary_scaler_without_state_leak(model) as (
        unscaled_model,
        scaler,
    ):
        for batch_index, start in enumerate(
            range(0, len(features), batch_size),
            start=1,
        ):
            stop = min(start + batch_size, len(features))
            batch_features = features[start:stop].to(device)
            batch_events = (
                event_ids[start:stop].to(device)
                if event_ids is not None
                else None
            )
            raw_input = _model_input(model, batch_features, batch_events)
            scaled_input = scaler(raw_input)
            batch_first, batch_second = _mean_absolute_taylor_sums(
                unscaled_model,
                scaled_input,
                coefficient_indices,
                max_order,
            )
            first_sum += batch_first
            second_sum += batch_second
            now = time.monotonic()
            if progress_label and (
                batch_index == total_batches
                or now - last_progress_at >= progress_interval_seconds
            ):
                elapsed = now - started_at
                batches_per_second = batch_index / elapsed if elapsed else 0.0
                remaining = (
                    (total_batches - batch_index) / batches_per_second
                    if batches_per_second
                    else 0.0
                )
                logger.info(
                    "%s: batch %d/%d (%.1f%%), elapsed %.1f min, ETA %.1f min",
                    progress_label,
                    batch_index,
                    total_batches,
                    100.0 * batch_index / total_batches,
                    elapsed / 60.0,
                    remaining / 60.0,
                )
                last_progress_at = now

    if len(features) == 0:
        raise ValueError("Cannot calculate Taylor coefficients for no events")

    coefficients: CoefficientDict = {
        "first_order": {
            name: float(first_sum[index] / len(features))
            for index, name in enumerate(feature_names)
        },
        "second_order": {},
    }
    if max_order >= 2:
        coefficients["second_order"] = {
            f"{feature_names[i]},{feature_names[j]}": float(
                second_sum[i, j] / len(features)
            )
            for i in range(len(feature_names))
            for j in range(i, len(feature_names))
        }
    if progress_label:
        logger.info(
            "%s: Taylor calculation finished in %.1f min",
            progress_label,
            (time.monotonic() - started_at) / 60.0,
        )
    return coefficients


def _set_grouped_tca_mode(model: t.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "_tca_mode"):
            module._tca_mode = enabled


@contextmanager
def _manually_extracted_scaler(
    model: t.nn.Module,
) -> Iterable[Tuple[t.nn.Module, Callable[[t.Tensor], t.Tensor]]]:
    """Independent reference implementation of scaler extraction for tests."""
    model = deepcopy(model)
    originals = {}

    def scale_dnn(dnn: t.nn.Module, values: t.Tensor) -> t.Tensor:
        return (
            values - dnn._scaler_shift.to(values.device)
        ) / dnn._scaler_scale.to(values.device)

    def scale(current: t.nn.Module, values: t.Tensor) -> t.Tensor:
        if isinstance(current, FoldCombinedDNN):
            even = values[0].long() % 2 == 0
            raw_features = values[1:].T
            scaled_even = scale(current.even_model, raw_features)
            scaled_odd = scale(current.odd_model, raw_features)
            result = values.clone()
            result[1:] = t.where(
                even.unsqueeze(1),
                scaled_even,
                scaled_odd,
            ).T
            return result
        if isinstance(current, GroupedDNN):
            result = t.zeros_like(values)
            assigned = t.zeros(len(values), dtype=t.bool, device=values.device)
            for conditions, payload in current._logic_pipeline:
                mask = t.ones(len(values), dtype=t.bool, device=values.device)
                for column, bounds in conditions:
                    if len(bounds) == 1:
                        target = values[:, column].new_tensor(float(bounds[0]))
                        mask &= t.isclose(
                            values[:, column],
                            target,
                            atol=1e-4,
                            rtol=0.0,
                        )
                    else:
                        mask &= values[:, column] >= bounds[0]
                        if bounds[1] != float("inf"):
                            mask &= values[:, column] <= bounds[1]
                result[mask] = scale(payload, values[mask])
                assigned |= mask
            result[~assigned] = scale(current._fallback_payload, values[~assigned])
            return result
        if hasattr(current, "wrapped_model"):
            indices = getattr(current, "indices", None)
            selected = values[:, indices] if indices is not None else values
            scaled = scale(current.wrapped_model, selected)
            if indices is None:
                return scaled
            result = values.clone()
            result[:, indices] = scaled
            return result
        return scale_dnn(current, values)

    try:
        scaler = lambda values: scale(model, values)
        _set_grouped_tca_mode(model, True)
        for name, buffer in model.named_buffers():
            if name.endswith("_scaler_shift"):
                originals[name] = buffer.clone()
                buffer.zero_()
            elif name.endswith("_scaler_scale"):
                originals[name] = buffer.clone()
                buffer.fill_(1.0)
        yield model, scaler
    finally:
        _set_grouped_tca_mode(model, False)


@contextmanager
def _temporary_scaler_without_state_leak(
    model: t.nn.Module,
) -> Iterable[Tuple[t.nn.Module, Callable[[t.Tensor], t.Tensor]]]:
    """Use the existing helper while restoring its grouped-model mode locally."""
    tca_modes = {
        module: module._tca_mode
        for module in model.modules()
        if hasattr(module, "_tca_mode")
    }
    try:
        with temporary_extract_scaler_callable(model) as extracted:
            yield extracted
    finally:
        for module, tca_mode in tca_modes.items():
            module._tca_mode = tca_mode


def calculate_taylor_coefficients_manually(
    model: Model,
    features: Union[np.ndarray, t.Tensor],
    feature_names: Sequence[str],
    *,
    event_ids: Optional[Union[np.ndarray, t.Tensor]] = None,
    max_order: int = 2,
    batch_size: int = 1024,
) -> CoefficientDict:
    """Reference calculation matching the notebook's hand-extracted scaler."""
    model_copy = deepcopy(model).eval()
    features = t.as_tensor(features, dtype=t.float32)
    events = (
        t.as_tensor(event_ids, dtype=t.float32)
        if event_ids is not None
        else None
    )
    indices = _feature_indices(model_copy, len(feature_names))
    first_sum = np.zeros(len(feature_names), dtype=np.float64)
    second_sum = np.zeros((len(feature_names), len(feature_names)))

    with _manually_extracted_scaler(model_copy) as (unscaled_model, scaler):
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            raw_input = _model_input(
                model_copy,
                features[start:stop],
                events[start:stop] if events is not None else None,
            )
            first, second = _mean_absolute_taylor_sums(
                unscaled_model,
                scaler(raw_input),
                indices,
                max_order,
            )
            first_sum += first
            second_sum += second

    result: CoefficientDict = {
        "first_order": {
            name: float(first_sum[i] / len(features))
            for i, name in enumerate(feature_names)
        },
        "second_order": {},
    }
    if max_order >= 2:
        result["second_order"] = {
            f"{feature_names[i]},{feature_names[j]}": float(
                second_sum[i, j] / len(features)
            )
            for i in range(len(feature_names))
            for j in range(i, len(feature_names))
        }
    return result


def _plot_coefficients(
    coefficients: Mapping[str, Mapping[str, float]],
    output_path: Path,
    *,
    top_n: int,
    second_order_only: bool,
    title: str,
    grouping: Optional[str] = None,
) -> None:
    entries = []
    if not second_order_only:
        entries.extend(
            (name, value, "first order")
            for name, value in coefficients["first_order"].items()
        )
    entries.extend(
        (name, value, "second order")
        for name, value in coefficients["second_order"].items()
    )
    entries = sorted(entries, key=lambda item: item[1], reverse=True)[:top_n]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_height = max(4.0, 0.45 * len(entries) + 1.5)
    fig, axis = plt.subplots(figsize=(10, figure_height))
    labels = [entry[0] for entry in entries][::-1]
    values = [entry[1] for entry in entries][::-1]
    color = TAYLOR_GROUPING_COLORS.get(grouping, DEFAULT_TAYLOR_COLOR)
    axis.barh(labels, values, color=color)
    axis.set_xlabel("mean absolute Taylor coefficient")
    axis.set_title(title)
    for tick_label in axis.get_yticklabels():
        tick_label.set_fontweight("bold")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_taylor_artifacts(
    coefficients: CoefficientDict,
    output_dir: Union[str, Path],
    *,
    top_n: int = 10,
    title: str = "Taylor coefficients",
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "taylor_coefficients.json",
        "metadata": output_dir / "metadata.json",
        "combined_png": output_dir / "taylor_coefficients_top_first_second.png",
        "combined_pdf": output_dir / "taylor_coefficients_top_first_second.pdf",
        "second_png": output_dir / "taylor_coefficients_top_second_order.png",
        "second_pdf": output_dir / "taylor_coefficients_top_second_order.pdf",
        "style": output_dir / ".plot_style_v2_grouping_colors",
    }
    paths["json"].write_text(
        json.dumps(_legacy_coefficient_dict(coefficients), indent=2) + "\n"
    )
    paths["metadata"].write_text(json.dumps(dict(metadata or {}), indent=2) + "\n")
    grouping = (metadata or {}).get("analysis_label")
    for extension in ("png", "pdf"):
        _plot_coefficients(
            coefficients,
            paths[f"combined_{extension}"],
            top_n=top_n,
            second_order_only=False,
            title=f"{title}: first and second order",
            grouping=grouping,
        )
        _plot_coefficients(
            coefficients,
            paths[f"second_{extension}"],
            top_n=top_n,
            second_order_only=True,
            title=f"{title}: second order",
            grouping=grouping,
        )
    paths["style"].write_text(
        "Taylor coefficient plots use grouping colors and bold labels.\n"
    )
    return paths


def _coefficient_dict_from_legacy(
    flat_coefficients: Mapping[str, float],
) -> CoefficientDict:
    return {
        "first_order": {
            name: value
            for name, value in flat_coefficients.items()
            if "," not in name
        },
        "second_order": {
            name: value
            for name, value in flat_coefficients.items()
            if "," in name
        },
    }


def rewrite_taylor_plots(
    output_dir: Union[str, Path],
    *,
    top_n: Optional[int] = None,
) -> Dict[str, Path]:
    """Regenerate Taylor coefficient plots from existing JSON artifacts."""
    output_dir = Path(output_dir)
    coefficient_path = output_dir / "taylor_coefficients.json"
    metadata_path = output_dir / "metadata.json"
    if not coefficient_path.is_file():
        raise FileNotFoundError(
            f"Missing Taylor coefficient JSON: {coefficient_path}"
        )

    metadata = (
        json.loads(metadata_path.read_text())
        if metadata_path.is_file()
        else {}
    )
    coefficients = _coefficient_dict_from_legacy(
        json.loads(coefficient_path.read_text())
    )
    return write_taylor_artifacts(
        coefficients,
        output_dir,
        top_n=top_n if top_n is not None else int(metadata.get("top_n", 10)),
        title=(
            f"{metadata.get('model_type', 'Taylor')} "
            f"{metadata.get('process', '')} "
            f"({metadata.get('analysis_label', '')}, "
            f"{metadata.get('category', '')})"
        ),
        metadata=metadata,
    )


def _analysis_frame(
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    training_variables: Sequence[str],
    process: str,
) -> np.ndarray:
    data = load_data(data_path, masks_path)
    process_view = data.ttbar if process == "ttbar" else data.data
    signal = getattr(process_view, f"SR_like_{process}").events
    background = getattr(process_view, f"AR_like_{process}").events
    frame = np.concatenate((
        signal[["event", *training_variables]].to_numpy(dtype=np.float32),
        background[["event", *training_variables]].to_numpy(dtype=np.float32),
    ))
    finite = np.isfinite(frame).all(axis=1)
    if not finite.all():
        logger.warning(
            "%s: dropping %d/%d rows with non-finite Taylor inputs",
            process,
            int((~finite).sum()),
            len(frame),
        )
    frame = frame[finite]
    if len(frame) == 0:
        raise ValueError(f"No finite events available for process {process}")
    return frame


def _category_frame(
    frame: np.ndarray,
    training_variables: Sequence[str],
    category: str,
) -> np.ndarray:
    try:
        selection = TAYLOR_CATEGORY_SELECTIONS[category]
    except KeyError as error:
        raise ValueError(f"Unsupported Taylor category: {category}") from error

    if selection is None:
        return frame

    column, operation, value = selection
    try:
        values = frame[:, 1 + list(training_variables).index(column)]
    except ValueError as error:
        raise KeyError(f"Missing Taylor-category column: {column}") from error
    if operation == "eq":
        mask = values == value
    elif operation == "ge":
        mask = values >= value
    elif operation == "isin":
        mask = np.isin(values, value)
    else:
        raise ValueError(f"Unsupported Taylor-category operation: {operation}")
    return frame[mask]


def _analysis_arrays(
    frame: np.ndarray,
    training_variables: Sequence[str],
    category: str = "inclusive",
) -> Tuple[np.ndarray, np.ndarray]:
    category_data = _category_frame(frame, training_variables, category)
    if len(category_data) == 0:
        raise ValueError(f"No events available for Taylor category {category}")
    return (
        category_data[:, 1:],
        category_data[:, 0],
    )


def run_taylor_coefficient_categories(
    *,
    even_model_path: Union[str, Path],
    odd_model_path: Union[str, Path],
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    training_var_path: Union[str, Path],
    process: str,
    output_dirs: Mapping[str, Union[str, Path]],
    analysis_label: str,
    model_type: str,
    max_order: int = 2,
    top_n: int = 10,
    batch_size: int = 1024,
) -> Dict[str, Dict[str, Path]]:
    """Calculate Taylor coefficients for multiple event categories efficiently."""
    training_variables = load_variables(training_var_path)
    frame = _analysis_frame(
        data_path,
        masks_path,
        training_variables,
        process,
    )
    model = load_fold_combined_model(
        Path(even_model_path),
        Path(odd_model_path),
    ).eval()

    results = {}
    for category, output_dir in output_dirs.items():
        features, event_ids = _analysis_arrays(
            frame,
            training_variables,
            category,
        )
        coefficients = calculate_taylor_coefficients(
            model,
            features,
            training_variables,
            event_ids=event_ids,
            max_order=max_order,
            batch_size=batch_size,
        )
        results[category] = write_taylor_artifacts(
            coefficients,
            output_dir,
            top_n=top_n,
            title=f"{model_type} {process} ({analysis_label}, {category})",
            metadata={
                "model_type": model_type,
                "process": process,
                "analysis_label": analysis_label,
                "category": category,
                "n_events": len(features),
                "max_order": max_order,
                "top_n": top_n,
                "even_model_path": str(even_model_path),
                "odd_model_path": str(odd_model_path),
            },
        )
    return results


def _write_comparison_method(
    coefficients: CoefficientDict,
    output_dir: Path,
    filename_prefix: str,
    method: str,
    top_n: int,
    grouping: str,
) -> None:
    (
        output_dir / f"{filename_prefix}taylor_coefficients_{method}.json"
    ).write_text(
        json.dumps(_legacy_coefficient_dict(coefficients), indent=2) + "\n"
    )
    for extension in ("png", "pdf"):
        _plot_coefficients(
            coefficients,
            output_dir
            / f"{filename_prefix}taylor_coefficients_{method}.{extension}",
            top_n=top_n,
            second_order_only=False,
            title=f"{filename_prefix.rstrip('_')} {method.replace('_', ' ')}",
            grouping=grouping,
        )


def run_taylor_coefficient_comparison(
    *,
    model_path: Union[str, Path],
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    training_var_path: Union[str, Path],
    output_dir: Union[str, Path],
    grouping: str,
    model_type: str,
    filename_prefix: str = "",
    max_order: int = 2,
    top_n: int = 10,
    batch_size: int = 1024,
) -> None:
    """Write extracted-scaler and hand-extracted results for one fold model."""
    variables = load_variables(training_var_path)
    frame = _analysis_frame(
        data_path,
        masks_path,
        variables,
        "wjets",
    )
    features, _ = _analysis_arrays(
        frame,
        variables,
    )
    model = load_model(Path(model_path)).eval()
    extracted = calculate_taylor_coefficients(
        model,
        features,
        variables,
        max_order=max_order,
        batch_size=batch_size,
    )
    manual = calculate_taylor_coefficients_manually(
        model,
        features,
        variables,
        max_order=max_order,
        batch_size=batch_size,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_comparison_method(
        extracted, output_dir, filename_prefix, "new_way", top_n, grouping
    )
    _write_comparison_method(
        manual, output_dir, filename_prefix, "notebook_way", top_n, grouping
    )
    differences = [
        abs(extracted[order][key] - manual[order][key])
        for order in extracted
        for key in extracted[order]
    ]
    metadata = {
        "model_type": model_type,
        "grouping": grouping,
        "model_path": str(model_path),
        "max_abs_difference": max(differences, default=0.0),
    }
    (output_dir / f"{filename_prefix}metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
