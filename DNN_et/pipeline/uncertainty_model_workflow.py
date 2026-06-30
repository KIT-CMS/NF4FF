import json
import logging
import random
import time
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from classes import (
    DNN,
    EnsembleStatUncWrapper,
    FoldCombinedDNN,
    GroupedDNN,
    create_training_dataset,
    load_data,
    load_fold_combined_model,
    save_model,
    train_dnn,
)
from groupings import grouping_bounds, grouping_source
from taylor_coefficient_analysis import (
    _analysis_arrays,
    _analysis_frame,
    calculate_taylor_coefficients,
)

logger = logging.getLogger(__name__)

SEED_START = 100
SEED_END = 199
UNCERTAINTY_GROUPINGS = ("njets",)
PROCESSES = ("wjets", "qcd", "ttbar")


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


def _load_config(path):
    with Path(path).open() as stream:
        return _from_dict(yaml.safe_load(stream), Config)


def _load_variables(path):
    with Path(path).open() as stream:
        return yaml.safe_load(stream).get("variables", [])


def _process_frames(data, process):
    process_view = data.ttbar if process == "ttbar" else data.data
    signal = getattr(process_view, f"SR_like_{process}").events.copy()
    background = getattr(process_view, f"AR_like_{process}").events.copy()
    weight_column = {
        "wjets": "weight_wjets",
        "qcd": "weight_qcd",
        "ttbar": "weight",
    }[process]
    return signal, background, weight_column


def _prepare_process_frames(
    data,
    process,
    grouping_name,
    training_variables,
    reduced_weight_dir,
):
    if process != "ttbar":
        feature_path = (
            Path(reduced_weight_dir)
            / process
            / f"reduced_weight_{grouping_name}.feather"
        )
        data.load_feature_file(feature_path)

    signal, background, weight_column = _process_frames(data, process)
    if process == "ttbar":
        source_weight = weight_column
    else:
        source_weight = (
            f"reduced_weight_{process}_{grouping_name}_nominal"
        )
        signal[weight_column] = signal[source_weight]
        background[weight_column] = background[source_weight]

    required_columns = [*training_variables, weight_column]
    prepared = []
    for label, frame in (("signal", signal), ("background", background)):
        finite = np.isfinite(
            frame[required_columns].to_numpy(dtype=np.float64)
        ).all(axis=1)
        dropped = int((~finite).sum())
        if dropped:
            logger.warning(
                "%s/%s/%s: dropping %d/%d rows with non-finite inputs",
                grouping_name,
                process,
                label,
                dropped,
                len(frame),
            )
        frame = frame.loc[finite].copy()
        if frame.empty:
            raise ValueError(
                f"{grouping_name}/{process}/{label}: no finite rows remain"
            )
        prepared.append(frame)
    return *prepared, weight_column


def _train_fold(
    config,
    grouping,
    training_variables,
    signal,
    background,
    weight_column,
    device,
    checkpoint_dir,
):
    train, validation = create_training_dataset(
        df_sig=signal,
        df_bkg=background,
        training_var=training_variables,
        weight_column=weight_column,
        balance=True,
        test_size=0.25,
        random_state=42,
    )
    base_model = DNN(
        input_nodes=train.X.shape[1],
        hidden_nodes=config.model.hidden_nodes,
        output_nodes=1,
        activation=config.model.activation,
        output_activation=config.model.output_activation,
        dropout=config.model.dropout,
        input_names=training_variables,
    )
    base_model.initialize_scaler(
        shift=train.X.mean(dim=0),
        scale=train.X.std(dim=0) + 1e-6,
    )
    model = GroupedDNN(grouping=grouping, default_model=base_model)
    model, _ = train_dnn(
        model=model,
        train=train,
        val=validation,
        epochs=config.training.epochs,
        lr=config.training.lr,
        loss_fn=None,
        device=device,
        checkpoint_dir=checkpoint_dir,
        scheduler_patience=config.scheduler.patience,
        early_stopping_patience=config.scheduler.early_stopping_patience,
        scheduler_factor=config.scheduler.factor,
        min_delta=config.scheduler.min_delta,
        min_lr=config.scheduler.min_lr,
    )
    return model


def train_uncertainty_models(
    *,
    data_path,
    masks_path,
    training_var_path,
    nn_config_path,
    output_dir,
    reduced_weight_dir,
    seed_start=SEED_START,
    seed_end=SEED_END,
):
    """Train 100 full-dataset models for the njets grouping."""
    if seed_end < seed_start:
        raise ValueError("seed_end must be greater than or equal to seed_start")
    if seed_end - seed_start + 1 != 100:
        raise ValueError("The njets uncertainty study requires exactly 100 seeds")

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    t.set_num_threads(8)
    config = _load_config(nn_config_path)
    training_variables = _load_variables(training_var_path)
    data = load_data(data_path, masks_path)
    output_dir = Path(output_dir)

    grouping_indices = {
        name: training_variables.index(grouping_source(name))
        for name in UNCERTAINTY_GROUPINGS
    }
    process_frames = {
        process: _prepare_process_frames(
            data,
            process,
            "njets",
            training_variables,
            reduced_weight_dir,
        )
        for process in PROCESSES
    }
    seeds = list(range(seed_start, seed_end + 1))
    for seed in seeds:
        t.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        logger.info("Training uncertainty ensemble seed %d", seed)

        for grouping_name in UNCERTAINTY_GROUPINGS:
            for process in PROCESSES:
                signal, background, weight_column = process_frames[process]
                signal_even = signal[signal["event"] % 2 == 0]
                signal_odd = signal[signal["event"] % 2 == 1]
                background_even = background[background["event"] % 2 == 0]
                background_odd = background[background["event"] % 2 == 1]
                grouping = {
                    grouping_indices[grouping_name]: grouping_bounds(grouping_name)
                }
                model_dir = output_dir / grouping_name / process / str(seed)

                even_model = _train_fold(
                    config,
                    grouping,
                    training_variables,
                    signal_odd,
                    background_odd,
                    weight_column,
                    device,
                    model_dir / "training_fold_even",
                )
                odd_model = _train_fold(
                    config,
                    grouping,
                    training_variables,
                    signal_even,
                    background_even,
                    weight_column,
                    device,
                    model_dir / "training_fold_odd",
                )
                combined_model = FoldCombinedDNN(
                    even_model=even_model,
                    odd_model=odd_model,
                    fold_id_name="event",
                )
                save_model(even_model, model_dir / "fold_even")
                save_model(odd_model, model_dir / "fold_odd")
                save_model(combined_model, model_dir)

    manifest = {
        "dataset_size": "full",
        "seed_start": seed_start,
        "seed_end": seed_end,
        "n_seeds": len(seeds),
        "groupings": list(UNCERTAINTY_GROUPINGS),
        "processes": list(PROCESSES),
        "combined_models_trained": (
            len(seeds) * len(UNCERTAINTY_GROUPINGS) * len(PROCESSES)
        ),
        "fold_models_trained": len(seeds) * len(UNCERTAINTY_GROUPINGS)
        * len(PROCESSES) * 2,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def summarize_taylor_ensemble(
    coefficients_by_seed: Mapping[str, Mapping[str, Mapping[str, float]]],
    top_n=None,
):
    entries = {}
    for seed_coefficients in coefficients_by_seed.values():
        for order, coefficients in seed_coefficients.items():
            for name, value in coefficients.items():
                key = f"{order}:{name}"
                entry = entries.setdefault(
                    key,
                    {"name": name, "order": order, "values": []},
                )
                entry["values"].append(float(value))

    summary = []
    expected_count = len(coefficients_by_seed)
    for entry in entries.values():
        if len(entry["values"]) != expected_count:
            raise ValueError(
                f"Coefficient {entry['name']} is missing from one or more seeds"
            )
        values = np.asarray(entry.pop("values"), dtype=np.float64)
        summary.append({
            **entry,
            "mean": float(values.mean()),
            "std": float(values.std()),
        })
    summary = sorted(summary, key=lambda item: item["mean"], reverse=True)
    return summary if top_n is None else summary[:top_n]


def _normalize_taylor_coefficients_to_max(
    coefficients_by_seed: Mapping[str, Mapping[str, Mapping[str, float]]],
):
    normalized = {}
    for seed, seed_coefficients in coefficients_by_seed.items():
        maximum = max(
            (
                abs(float(value))
                for coefficients in seed_coefficients.values()
                for value in coefficients.values()
            ),
            default=0.0,
        )
        if maximum <= 0.0:
            scale = 1.0
        else:
            scale = maximum
        normalized[str(seed)] = {
            order: {
                name: float(value) / scale
                for name, value in coefficients.items()
            }
            for order, coefficients in seed_coefficients.items()
        }
    return normalized


def summarize_taylor_ensemble_normalized_to_max(
    coefficients_by_seed: Mapping[str, Mapping[str, Mapping[str, float]]],
    top_n=None,
):
    return summarize_taylor_ensemble(
        _normalize_taylor_coefficients_to_max(coefficients_by_seed),
        top_n=top_n,
    )


def _sigma_label(std_scale):
    if np.isclose(std_scale, 1.0):
        return r"$\pm 1\sigma$"
    if np.isclose(std_scale, 0.5):
        return r"$\pm 0.5\sigma$"
    return rf"$\pm {std_scale:g}\sigma$"


def plot_taylor_ensemble(summary, output_path, *, title, std_scale=1.0):
    """Plot only each coefficient's mean ± one standard deviation interval."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = list(reversed(summary))
    positions = np.arange(len(entries))
    means = np.asarray([entry["mean"] for entry in entries])
    stds = std_scale * np.asarray([entry["std"] for entry in entries])
    left = means - stds
    widths = 2.0 * stds
    colors = [
        "#d95f02" if entry["order"] == "second_order" else "#1b9e77"
        for entry in entries
    ]

    figure_height = max(4.0, 0.5 * len(entries) + 1.5)
    fig, axis = plt.subplots(figsize=(10, figure_height))
    axis.barh(
        positions,
        widths,
        left=left,
        height=0.34,
        color=colors,
        alpha=0.55,
    )
    axis.scatter(
        means,
        positions,
        marker="|",
        s=180,
        linewidths=2.5,
        color="#202020",
        zorder=3,
    )
    axis.set_yticks(positions, [entry["name"] for entry in entries])
    axis.set_xlabel("mean absolute Taylor coefficient")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    orders = {entry["order"] for entry in entries}
    sigma_label = _sigma_label(std_scale)
    legend_handles = []
    if "first_order" in orders:
        legend_handles.append(
            Patch(
                color="#1b9e77",
                alpha=0.55,
                label=f"first order {sigma_label}",
            )
        )
    if "second_order" in orders:
        legend_handles.append(
            Patch(
                color="#d95f02",
                alpha=0.55,
                label=f"second order {sigma_label}",
            )
        )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#202020",
            marker="|",
            markersize=12,
            markeredgewidth=2.5,
            linestyle="None",
            label="mean",
        )
    )
    axis.legend(handles=legend_handles)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _set_dropout_mask(model, mask_index):
    found = 0
    for module in model.modules():
        if hasattr(module, "active_mask") and hasattr(module, "masks"):
            module.active_mask = mask_index
            found += 1
    if found == 0:
        raise ValueError("The selected model has no non-zero dropout layers")


def calculate_dropout_mask_taylor_coefficients(
    model,
    features,
    feature_names,
    event_ids,
    *,
    n_masks=100,
    max_order=2,
    batch_size=1024,
    random_seed=42,
    progress_prefix="dropout masks",
):
    """Calculate Taylor coefficients for fixed masks of one trained model."""
    t.manual_seed(random_seed)
    wrapper = EnsembleStatUncWrapper(
        model=model,
        ensemble_size=n_masks,
        direction="Nominal",
    ).eval()
    masked_model = wrapper.wrapped_model
    coefficients = {}
    try:
        for mask_index in range(1, n_masks + 1):
            _set_dropout_mask(masked_model, mask_index)
            coefficients[str(mask_index)] = calculate_taylor_coefficients(
                masked_model,
                features,
                feature_names,
                event_ids=event_ids,
                max_order=max_order,
                batch_size=batch_size,
                progress_label=(
                    f"{progress_prefix}: mask {mask_index}/{n_masks}"
                ),
            )
    finally:
        _set_dropout_mask(masked_model, None)
    return coefficients


def plot_taylor_method_comparison(
    model_summary,
    dropout_summary,
    output_path,
    *,
    top_n,
    title,
    std_scale=1.0,
):
    """Compare seeded-model and dropout-mask mean ± std intervals."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dropout_by_key = {
        (entry["order"], entry["name"]): entry
        for entry in dropout_summary
    }
    selected_models = model_summary[:top_n]
    selected_dropout = [
        dropout_by_key[(entry["order"], entry["name"])]
        for entry in selected_models
    ]
    selected_models = list(reversed(selected_models))
    selected_dropout = list(reversed(selected_dropout))
    positions = np.arange(len(selected_models))

    fig, axis = plt.subplots(
        figsize=(11, max(4.0, 0.58 * len(selected_models) + 1.5))
    )
    for entries, offset, color, label in (
        (selected_models, -0.12, "#377eb8", "100 trained models"),
        (selected_dropout, 0.12, "#e41a1c", "100 dropout masks"),
    ):
        means = np.asarray([entry["mean"] for entry in entries])
        stds = std_scale * np.asarray([entry["std"] for entry in entries])
        axis.barh(
            positions + offset,
            2.0 * stds,
            left=means - stds,
            height=0.20,
            color=color,
            alpha=0.5,
            label=label,
        )
        axis.scatter(
            means,
            positions + offset,
            marker="|",
            s=130,
            linewidths=2.0,
            color=color,
            zorder=3,
        )

    axis.set_yticks(positions, [entry["name"] for entry in selected_models])
    axis.set_xlabel("mean absolute Taylor coefficient")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def uncertainty_taylor_coefficient_paths(output_dir, process):
    result_dir = Path(output_dir) / process
    return {
        "models_coefficients": result_dir / "trained_models_coefficients.json",
        "dropout_coefficients": result_dir / "dropout_mask_coefficients.json",
    }


def uncertainty_taylor_plot_paths(output_dir, process, top_n):
    result_dir = Path(output_dir) / process
    return {
        "models_summary": result_dir / f"trained_models_top_{top_n}.json",
        "dropout_summary": result_dir / f"dropout_masks_top_{top_n}.json",
        "models_png": result_dir / f"trained_models_top_{top_n}.png",
        "models_pdf": result_dir / f"trained_models_top_{top_n}.pdf",
        "dropout_png": result_dir / f"dropout_masks_top_{top_n}.png",
        "dropout_pdf": result_dir / f"dropout_masks_top_{top_n}.pdf",
        "comparison_png": result_dir / f"method_comparison_top_{top_n}.png",
        "comparison_pdf": result_dir / f"method_comparison_top_{top_n}.pdf",
    }


def uncertainty_taylor_normalized_plot_paths(output_dir, process, top_n):
    result_dir = Path(output_dir) / process / "normalized_to_max"
    return {
        "models_summary": result_dir / f"trained_models_top_{top_n}.json",
        "dropout_summary": result_dir / f"dropout_masks_top_{top_n}.json",
        "models_png": result_dir / f"trained_models_top_{top_n}.png",
        "models_pdf": result_dir / f"trained_models_top_{top_n}.pdf",
        "dropout_png": result_dir / f"dropout_masks_top_{top_n}.png",
        "dropout_pdf": result_dir / f"dropout_masks_top_{top_n}.pdf",
        "comparison_png": result_dir / f"method_comparison_top_{top_n}.png",
        "comparison_pdf": result_dir / f"method_comparison_top_{top_n}.pdf",
    }


def _taylor_order_key(taylor_order):
    if int(taylor_order) == 1:
        return "first_order"
    if int(taylor_order) == 2:
        return "second_order"
    raise ValueError(f"taylor_order must be 1 or 2, got {taylor_order}")


def _taylor_order_label(taylor_order):
    return {
        1: "first-order",
        2: "second-order",
    }[int(taylor_order)]


def uncertainty_taylor_normalized_single_order_plot_paths(
    output_dir,
    process,
    top_n,
    taylor_order,
):
    result_dir = (
        Path(output_dir)
        / process
        / "normalized_to_max"
        / f"order_{int(taylor_order)}_only"
    )
    return {
        "models_summary": result_dir / f"trained_models_top_{top_n}.json",
        "dropout_summary": result_dir / f"dropout_masks_top_{top_n}.json",
        "models_png": result_dir / f"trained_models_top_{top_n}.png",
        "models_pdf": result_dir / f"trained_models_top_{top_n}.pdf",
        "dropout_png": result_dir / f"dropout_masks_top_{top_n}.png",
        "dropout_pdf": result_dir / f"dropout_masks_top_{top_n}.pdf",
        "comparison_png": result_dir / f"method_comparison_top_{top_n}.png",
        "comparison_pdf": result_dir / f"method_comparison_top_{top_n}.pdf",
    }


def uncertainty_taylor_artifact_paths(output_dir, process, top_n):
    return {
        **uncertainty_taylor_coefficient_paths(output_dir, process),
        **uncertainty_taylor_plot_paths(output_dir, process, top_n),
    }


def uncertainty_taylor_normalized_artifact_paths(output_dir, process, top_n):
    return {
        **uncertainty_taylor_coefficient_paths(output_dir, process),
        **uncertainty_taylor_normalized_plot_paths(output_dir, process, top_n),
    }


def uncertainty_taylor_normalized_single_order_artifact_paths(
    output_dir,
    process,
    top_n,
    taylor_order,
):
    return {
        **uncertainty_taylor_coefficient_paths(output_dir, process),
        **uncertainty_taylor_normalized_single_order_plot_paths(
            output_dir,
            process,
            top_n,
            taylor_order,
        ),
    }


def analyze_uncertainty_model_taylor_process(
    *,
    process,
    models_dir,
    data_path,
    masks_path,
    training_var_path,
    output_dir,
    seed_start=SEED_START,
    seed_end=SEED_END,
    dropout_model_seed=SEED_START,
    n_dropout_masks=100,
    max_order=2,
    top_n=10,
    batch_size=4096,
    cpu_threads=8,
):
    """Calculate Taylor coefficients for one process."""
    if process not in PROCESSES:
        raise ValueError(f"Unsupported process: {process}")
    if seed_end - seed_start + 1 != 100:
        raise ValueError("The Taylor comparison requires exactly 100 trained models")
    if n_dropout_masks != 100:
        raise ValueError("The Taylor comparison requires exactly 100 dropout masks")
    if cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    t.set_num_threads(cpu_threads)
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    training_variables = _load_variables(training_var_path)
    seeds = list(range(seed_start, seed_end + 1))
    grouping_name = "njets"
    process_started_at = time.monotonic()
    logger.info(
        "Starting uncertainty Taylor calculation for %s: %d models and "
        "%d dropout masks, order=%d, batch_size=%d, "
        "device=cpu, CPU threads=%d",
        process,
        len(seeds),
        n_dropout_masks,
        max_order,
        batch_size,
        t.get_num_threads(),
    )
    frame = _analysis_frame(
        data_path,
        masks_path,
        training_variables,
        process,
    )
    features, event_ids = _analysis_arrays(frame, training_variables)
    logger.info(
        "%s: %d events and %d features",
        process,
        len(features),
        len(training_variables),
    )
    coefficients_by_seed: Dict[str, Dict[str, Dict[str, float]]] = {}
    dropout_source_model = None
    for seed_index, seed in enumerate(seeds, start=1):
        model_dir = models_dir / grouping_name / process / str(seed)
        model = load_fold_combined_model(
            model_dir / "fold_even",
            model_dir / "fold_odd",
        ).eval()
        coefficients_by_seed[str(seed)] = calculate_taylor_coefficients(
            model,
            features,
            training_variables,
            event_ids=event_ids,
            max_order=max_order,
            batch_size=batch_size,
            progress_label=(
                f"{process}: trained model {seed_index}/{len(seeds)} "
                f"(seed {seed})"
            ),
        )
        if seed == dropout_model_seed:
            dropout_source_model = model
        else:
            del model

    if dropout_source_model is None:
        raise ValueError(
            f"dropout_model_seed {dropout_model_seed} is outside the seed range"
        )
    dropout_coefficients = calculate_dropout_mask_taylor_coefficients(
        dropout_source_model,
        features,
        training_variables,
        event_ids,
        n_masks=n_dropout_masks,
        max_order=max_order,
        batch_size=batch_size,
        progress_prefix=process,
    )
    del dropout_source_model

    paths = uncertainty_taylor_coefficient_paths(output_dir, process)
    paths["models_coefficients"].parent.mkdir(parents=True, exist_ok=True)
    paths["models_coefficients"].write_text(
        json.dumps(coefficients_by_seed, indent=2) + "\n"
    )
    paths["dropout_coefficients"].write_text(
        json.dumps(dropout_coefficients, indent=2) + "\n"
    )
    logger.info(
        "Uncertainty Taylor calculation for %s finished in %.1f min",
        process,
        (time.monotonic() - process_started_at) / 60.0,
    )
    return {key: str(path) for key, path in paths.items()}


def plot_uncertainty_model_taylor_process(
    *,
    process,
    output_dir,
    top_n=10,
):
    """Summarize and plot previously calculated coefficients for one process."""
    if process not in PROCESSES:
        raise ValueError(f"Unsupported process: {process}")
    coefficient_paths = uncertainty_taylor_coefficient_paths(
        output_dir,
        process,
    )
    coefficients_by_seed = json.loads(
        coefficient_paths["models_coefficients"].read_text()
    )
    dropout_coefficients = json.loads(
        coefficient_paths["dropout_coefficients"].read_text()
    )
    model_summary = summarize_taylor_ensemble(coefficients_by_seed)
    dropout_summary = summarize_taylor_ensemble(dropout_coefficients)
    paths = uncertainty_taylor_plot_paths(output_dir, process, top_n)
    paths["models_summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["models_summary"].write_text(
        json.dumps(model_summary[:top_n], indent=2) + "\n"
    )
    paths["dropout_summary"].write_text(
        json.dumps(dropout_summary[:top_n], indent=2) + "\n"
    )
    for extension in ("png", "pdf"):
        plot_taylor_ensemble(
            model_summary[:top_n],
            paths[f"models_{extension}"],
            title=f"{process} njets: Taylor coefficients from 100 models",
        )
        plot_taylor_ensemble(
            dropout_summary[:top_n],
            paths[f"dropout_{extension}"],
            title=f"{process} njets: Taylor coefficients from 100 masks",
        )
        plot_taylor_method_comparison(
            model_summary,
            dropout_summary,
            paths[f"comparison_{extension}"],
            top_n=top_n,
            title=f"{process} njets: trained models vs dropout masks",
        )
    logger.info(
        "Wrote uncertainty Taylor summaries and plots for %s to %s",
        process,
        paths["models_summary"].parent,
    )
    return {key: str(path) for key, path in paths.items()}


def plot_uncertainty_model_taylor_process_normalized_to_max(
    *,
    process,
    output_dir,
    top_n=10,
):
    """Plot Taylor summaries after normalizing each model/mask to its max coefficient."""
    if process not in PROCESSES:
        raise ValueError(f"Unsupported process: {process}")
    coefficient_paths = uncertainty_taylor_coefficient_paths(
        output_dir,
        process,
    )
    coefficients_by_seed = json.loads(
        coefficient_paths["models_coefficients"].read_text()
    )
    dropout_coefficients = json.loads(
        coefficient_paths["dropout_coefficients"].read_text()
    )
    model_summary = summarize_taylor_ensemble_normalized_to_max(
        coefficients_by_seed
    )
    dropout_summary = summarize_taylor_ensemble_normalized_to_max(
        dropout_coefficients
    )
    paths = uncertainty_taylor_normalized_plot_paths(
        output_dir,
        process,
        top_n,
    )
    paths["models_summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["models_summary"].write_text(
        json.dumps(model_summary[:top_n], indent=2) + "\n"
    )
    paths["dropout_summary"].write_text(
        json.dumps(dropout_summary[:top_n], indent=2) + "\n"
    )
    for extension in ("png", "pdf"):
        plot_taylor_ensemble(
            model_summary[:top_n],
            paths[f"models_{extension}"],
            title=(
                f"{process} njets: Taylor coefficients from 100 models "
                "(normalized)"
            ),
        )
        plot_taylor_ensemble(
            dropout_summary[:top_n],
            paths[f"dropout_{extension}"],
            title=(
                f"{process} njets: Taylor coefficients from 100 masks "
                "(normalized)"
            ),
        )
        plot_taylor_method_comparison(
            model_summary,
            dropout_summary,
            paths[f"comparison_{extension}"],
            top_n=top_n,
            title=(
                f"{process} njets: trained models vs dropout masks "
                "(normalized)"
            ),
        )
    logger.info(
        "Wrote normalized uncertainty Taylor summaries and plots for %s to %s",
        process,
        paths["models_summary"].parent,
    )
    return {key: str(path) for key, path in paths.items()}


def _filter_summary_to_order(summary, taylor_order):
    order_key = _taylor_order_key(taylor_order)
    filtered = [entry for entry in summary if entry["order"] == order_key]
    if not filtered:
        raise ValueError(f"No {order_key} Taylor coefficients found")
    return filtered


def plot_uncertainty_model_taylor_process_normalized_single_order(
    *,
    process,
    output_dir,
    top_n=10,
    taylor_order=1,
):
    """Plot one Taylor order after normalizing each model/mask to its max coefficient."""
    if process not in PROCESSES:
        raise ValueError(f"Unsupported process: {process}")
    _taylor_order_key(taylor_order)
    coefficient_paths = uncertainty_taylor_coefficient_paths(
        output_dir,
        process,
    )
    coefficients_by_seed = json.loads(
        coefficient_paths["models_coefficients"].read_text()
    )
    dropout_coefficients = json.loads(
        coefficient_paths["dropout_coefficients"].read_text()
    )
    model_summary = _filter_summary_to_order(
        summarize_taylor_ensemble_normalized_to_max(coefficients_by_seed),
        taylor_order,
    )
    dropout_summary = _filter_summary_to_order(
        summarize_taylor_ensemble_normalized_to_max(dropout_coefficients),
        taylor_order,
    )
    order_label = _taylor_order_label(taylor_order)
    paths = uncertainty_taylor_normalized_single_order_plot_paths(
        output_dir,
        process,
        top_n,
        taylor_order,
    )
    paths["models_summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["models_summary"].write_text(
        json.dumps(model_summary[:top_n], indent=2) + "\n"
    )
    paths["dropout_summary"].write_text(
        json.dumps(dropout_summary[:top_n], indent=2) + "\n"
    )
    for extension in ("png", "pdf"):
        plot_taylor_ensemble(
            model_summary[:top_n],
            paths[f"models_{extension}"],
            title=(
                f"{process} njets: {order_label} Taylor coefficients "
                "from 100 models (normalized)"
            ),
        )
        plot_taylor_ensemble(
            dropout_summary[:top_n],
            paths[f"dropout_{extension}"],
            title=(
                f"{process} njets: {order_label} Taylor coefficients "
                "from 100 masks (normalized)"
            ),
        )
        plot_taylor_method_comparison(
            model_summary,
            dropout_summary,
            paths[f"comparison_{extension}"],
            top_n=top_n,
            title=(
                f"{process} njets: {order_label} trained models vs "
                "dropout masks (normalized)"
            ),
        )
    logger.info(
        "Wrote normalized %s uncertainty Taylor plots for %s to %s",
        order_label,
        process,
        paths["models_summary"].parent,
    )
    return {key: str(path) for key, path in paths.items()}


def write_uncertainty_taylor_manifest(
    *,
    output_dir,
    seed_start=SEED_START,
    seed_end=SEED_END,
    dropout_model_seed=SEED_START,
    n_dropout_masks=100,
    max_order=2,
    top_n=10,
    batch_size=4096,
    cpu_threads=8,
):
    output_dir = Path(output_dir)
    manifest = {
        "seed_start": seed_start,
        "seed_end": seed_end,
        "n_models_per_grouping_process": seed_end - seed_start + 1,
        "grouping": "njets",
        "dropout_model_seed": dropout_model_seed,
        "n_dropout_masks": n_dropout_masks,
        "max_order": max_order,
        "top_n": top_n,
        "batch_size": batch_size,
        "device": "cpu",
        "cpu_threads_per_process": cpu_threads,
        "artifacts": {
            process: {
                key: str(path)
                for key, path in uncertainty_taylor_artifact_paths(
                    output_dir,
                    process,
                    top_n,
                ).items()
            }
            for process in PROCESSES
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Wrote uncertainty Taylor manifest to %s", manifest_path)
    return manifest


def write_uncertainty_taylor_normalized_manifest(
    *,
    output_dir,
    seed_start=SEED_START,
    seed_end=SEED_END,
    dropout_model_seed=SEED_START,
    n_dropout_masks=100,
    max_order=2,
    top_n=10,
    batch_size=4096,
    cpu_threads=8,
):
    output_dir = Path(output_dir)
    manifest = {
        "seed_start": seed_start,
        "seed_end": seed_end,
        "n_models_per_grouping_process": seed_end - seed_start + 1,
        "grouping": "njets",
        "dropout_model_seed": dropout_model_seed,
        "n_dropout_masks": n_dropout_masks,
        "max_order": max_order,
        "top_n": top_n,
        "batch_size": batch_size,
        "device": "cpu",
        "cpu_threads_per_process": cpu_threads,
        "normalization": "Each trained model or dropout mask is divided by its largest Taylor coefficient before mean/std aggregation.",
        "artifacts": {
            process: {
                key: str(path)
                for key, path in uncertainty_taylor_normalized_artifact_paths(
                    output_dir,
                    process,
                    top_n,
                ).items()
            }
            for process in PROCESSES
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "normalized_to_max_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Wrote normalized uncertainty Taylor manifest to %s",
        manifest_path,
    )
    return manifest


def write_uncertainty_taylor_normalized_single_order_manifest(
    *,
    output_dir,
    seed_start=SEED_START,
    seed_end=SEED_END,
    dropout_model_seed=SEED_START,
    n_dropout_masks=100,
    max_order=2,
    top_n=10,
    taylor_order=1,
    batch_size=4096,
    cpu_threads=8,
):
    _taylor_order_key(taylor_order)
    output_dir = Path(output_dir)
    manifest = {
        "seed_start": seed_start,
        "seed_end": seed_end,
        "n_models_per_grouping_process": seed_end - seed_start + 1,
        "grouping": "njets",
        "dropout_model_seed": dropout_model_seed,
        "n_dropout_masks": n_dropout_masks,
        "max_order": max_order,
        "plotted_taylor_order": int(taylor_order),
        "top_n": top_n,
        "batch_size": batch_size,
        "device": "cpu",
        "cpu_threads_per_process": cpu_threads,
        "normalization": "Each trained model or dropout mask is divided by its largest Taylor coefficient before mean/std aggregation.",
        "artifacts": {
            process: {
                key: str(path)
                for key, path in uncertainty_taylor_normalized_single_order_artifact_paths(
                    output_dir,
                    process,
                    top_n,
                    taylor_order,
                ).items()
            }
            for process in PROCESSES
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        output_dir
        / f"normalized_to_max_order_{int(taylor_order)}_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Wrote normalized single-order uncertainty Taylor manifest to %s",
        manifest_path,
    )
    return manifest


def analyze_uncertainty_model_taylor_coefficients(
    *,
    models_dir,
    data_path,
    masks_path,
    training_var_path,
    output_dir,
    seed_start=SEED_START,
    seed_end=SEED_END,
    dropout_model_seed=SEED_START,
    n_dropout_masks=100,
    max_order=2,
    top_n=10,
    batch_size=4096,
    cpu_threads=8,
):
    """Backward-compatible sequential analysis across all processes."""
    for process in PROCESSES:
        analyze_uncertainty_model_taylor_process(
            process=process,
            models_dir=models_dir,
            data_path=data_path,
            masks_path=masks_path,
            training_var_path=training_var_path,
            output_dir=output_dir,
            seed_start=seed_start,
            seed_end=seed_end,
            dropout_model_seed=dropout_model_seed,
            n_dropout_masks=n_dropout_masks,
            max_order=max_order,
            top_n=top_n,
            batch_size=batch_size,
            cpu_threads=cpu_threads,
        )
        plot_uncertainty_model_taylor_process(
            process=process,
            output_dir=output_dir,
            top_n=top_n,
        )
    return write_uncertainty_taylor_manifest(
        output_dir=output_dir,
        seed_start=seed_start,
        seed_end=seed_end,
        dropout_model_seed=dropout_model_seed,
        n_dropout_masks=n_dropout_masks,
        max_order=max_order,
        top_n=top_n,
        batch_size=batch_size,
        cpu_threads=cpu_threads,
    )
