import argparse
from pathlib import Path
from typing import Union

import correctionlib as cr
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import yaml

from classes import load_data, load_model, load_variables
from classes.DataHandling import FeatureRegistry, FeatureStore
from ff_calculation import DEFAULT_PROCESS_FRACTIONS_PATH
from plotting import (
    CMS_CHANNEL_TITLE,
    CMS_CATEGORY_TITLE,
    CMS_LABEL,
    CMS_LUMI_TITLE,
    adjust_ylim_for_legend,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_ROOT = PROJECT_ROOT / "Law_workflow_results"
DATA_PATH = WORKFLOW_ROOT / "data" / "dataframe_complete.feather"
MASKS_PATH = PROJECT_ROOT / "configs" / "masks.yaml"
TRAINING_VAR_PATH = PROJECT_ROOT / "configs" / "training_variables.yaml"
PLOTTING_CONFIG_PATH = PROJECT_ROOT / "configs" / "plotting.yaml"
LABELS_CONFIG_PATH = PROJECT_ROOT / "configs" / "labels.yaml"
MODEL_DIR = WORKFLOW_ROOT / "training_fraction"
PLOTS_DIR = WORKFLOW_ROOT / "plots" / "training_fraction"
FEATURE_STORE_PATH = (
    WORKFLOW_ROOT
    / "data"
    / "features"
    / "training_fraction"
    / "process_fractions.feather"
)
FEATURE_REGISTRY_PATH = WORKFLOW_ROOT / "data" / "features" / "feature_registry.json"

OUTPUT_COLUMNS = (
    "fraction_qcd",
    "fraction_wjets",
    "fraction_ttbar",
)
CLASSIC_FRACTIONS = {
    "data": "QCD",
    "wjets": "Wjets",
    "ttbar": "ttbar",
}
NN_FRACTIONS = {
    "data": "fraction_qcd",
    "wjets": "fraction_wjets",
    "ttbar": "fraction_ttbar",
}
FRACTION_COLORS = {
    "data": "#b9ac70",
    "wjets": "#e76300",
    "ttbar": "#832db6",
}


def _read_yaml(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _read_labels_yaml(path):
    labels_by_channel = {}
    current_channel = None

    with open(path, "r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            indent = len(line) - len(line.lstrip(" "))

            if not stripped or stripped.startswith("#"):
                continue

            if stripped.endswith(":") and ":" not in stripped[:-1] and indent <= 1:
                current_channel = stripped[:-1]
                labels_by_channel.setdefault(current_channel, {})
                continue

            if current_channel is None or indent < 4:
                continue

            key_value = line.strip().split(":", 1)
            if len(key_value) == 2:
                key, value = key_value
                labels_by_channel[current_channel][key] = (
                    value.strip().strip('"').strip("'")
                )

    return labels_by_channel


def _get_bins(plotting_config, variable):
    bin_spec = plotting_config.get("bins_by_variable", {}).get(variable)
    if bin_spec is None:
        raise KeyError(f"No bin specification found for variable: {variable}")

    if isinstance(bin_spec, (list, tuple)) and len(bin_spec) == 3:
        start, stop, num = bin_spec
        return np.linspace(float(start), float(stop), int(num))

    return np.asarray(bin_spec, dtype=float)


def _get_label(labels_config, variable, channel="et"):
    labels_by_channel = labels_config.get(channel, {}) if isinstance(labels_config, dict) else {}
    return labels_by_channel.get(variable, variable)


def _predict_fraction_probabilities(
    model: t.nn.Module,
    frame,
    training_variables,
    *,
    device: t.device,
    batch_size: int,
) -> np.ndarray:
    probabilities = []

    model = model.to(device).eval()
    features = frame[training_variables].to_numpy(dtype=np.float32)
    parity = (frame["event"].to_numpy(dtype=np.int64) % 2).astype(np.float32)

    with t.no_grad():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            batch = np.concatenate(
                [
                    parity[start:stop].reshape(1, -1),
                    features[start:stop].T,
                ],
                axis=0,
            )
            batch_tensor = t.tensor(batch, dtype=t.float32, device=device)
            logits = model(batch_tensor)
            batch_probabilities = t.softmax(logits, dim=1)
            probabilities.append(batch_probabilities.cpu().numpy())

    return np.concatenate(probabilities, axis=0)


def calculate_and_store_fraction_nn_outputs(
    *,
    data_path: Union[str, Path] = DATA_PATH,
    masks_path: Union[str, Path] = MASKS_PATH,
    training_var_path: Union[str, Path] = TRAINING_VAR_PATH,
    model_dir: Union[str, Path] = MODEL_DIR,
    feature_store_path: Union[str, Path] = FEATURE_STORE_PATH,
    feature_registry_path: Union[str, Path] = FEATURE_REGISTRY_PATH,
    batch_size: int = 100_000,
) -> Path:
    frame = _load_frame_with_fraction_outputs(
        data_path=data_path,
        masks_path=masks_path,
        training_var_path=training_var_path,
        model_dir=model_dir,
        batch_size=batch_size,
    )
    return _store_fraction_output_frame(
        frame,
        feature_store_path=feature_store_path,
        feature_registry_path=feature_registry_path,
    )


def _store_fraction_output_frame(
    frame,
    *,
    feature_store_path: Union[str, Path],
    feature_registry_path: Union[str, Path],
) -> Path:

    feature_df = frame[["event"]].copy()
    feature_df.insert(0, "row_index", frame.index.to_numpy(dtype=np.int64))
    for column_name in OUTPUT_COLUMNS:
        feature_df[column_name] = frame[column_name].to_numpy(dtype=np.float32)

    registry = FeatureRegistry(feature_registry_path)
    store = FeatureStore(feature_store_path, registry)
    store.write(feature_df)
    store.save()
    registry.save()

    return Path(feature_store_path)


def _load_frame_with_fraction_outputs(
    *,
    data_path,
    masks_path,
    training_var_path,
    model_dir,
    batch_size,
):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    df = load_data(data_path, masks_path)
    frame = df.AR.events.copy()
    training_variables = load_variables(training_var_path)
    model = load_model(model_dir, device=str(device)).eval()

    probabilities = _predict_fraction_probabilities(
        model,
        frame,
        training_variables,
        device=device,
        batch_size=batch_size,
    )

    if probabilities.shape[1] != len(OUTPUT_COLUMNS):
        raise ValueError(
            "Expected three multiclass probabilities, got "
            f"shape {probabilities.shape}."
        )

    probability_sum = probabilities.sum(axis=1)
    if not np.allclose(probability_sum, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Fraction NN probabilities do not sum to one.")

    for column_index, column_name in enumerate(OUTPUT_COLUMNS):
        frame[column_name] = probabilities[:, column_index].astype(np.float32)

    return frame


def _binned_mean(values, fractions, bins):
    values = np.asarray(values, dtype=np.float64)
    fractions = np.asarray(fractions, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(fractions)
    values = values[finite]
    fractions = fractions[finite]

    bin_indices = np.digitize(values, bins) - 1
    bin_indices[values == bins[-1]] = len(bins) - 2

    means = np.full(len(bins) - 1, np.nan, dtype=np.float64)
    for bin_index in range(len(means)):
        in_bin = bin_indices == bin_index
        if in_bin.any():
            means[bin_index] = fractions[in_bin].mean(dtype=np.float64)

    return means


def _classic_process_fractions(frame, process_fractions):
    finite = np.isfinite(frame[["mt_1", "njets"]].to_numpy(dtype=np.float64)).all(axis=1)
    fractions = {
        key: np.full(len(frame), np.nan, dtype=np.float64)
        for key in CLASSIC_FRACTIONS
    }
    finite_frame = frame.loc[finite]

    for key, process_name in CLASSIC_FRACTIONS.items():
        fractions[key][finite] = process_fractions.evaluate(
            process_name,
            finite_frame["mt_1"].to_numpy(),
            finite_frame["njets"].to_numpy(),
            "nominal",
        )

    return fractions


def plot_fraction_comparisons(
    frame,
    *,
    output_dir: Union[str, Path] = PLOTS_DIR,
    process_fractions_path: Union[str, Path] = DEFAULT_PROCESS_FRACTIONS_PATH,
    plotting_config_path: Union[str, Path] = PLOTTING_CONFIG_PATH,
    labels_config_path: Union[str, Path] = LABELS_CONFIG_PATH,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotting_config = _read_yaml(plotting_config_path)
    labels_config = _read_labels_yaml(labels_config_path)
    variables = plotting_config.get("variables_set_small", [])
    if not variables:
        raise ValueError("No variables configured in variables_set_small.")

    process_fractions = cr.CorrectionSet.from_file(
        str(process_fractions_path)
    )["process_fractions"]
    classic_fractions = _classic_process_fractions(frame, process_fractions)
    stack_order = ("data", "wjets", "ttbar")
    stack_labels = {
        "data": "QCD",
        "wjets": "Wjets",
        "ttbar": "ttbar",
    }

    for variable in variables:
        bins = _get_bins(plotting_config, variable)

        fig, axis = plt.subplots(figsize=(8, 6))

        nn_cumulative = np.zeros(len(bins) - 1, dtype=np.float64)
        classic_cumulative = np.zeros(len(bins) - 1, dtype=np.float64)

        for stack_index, key in enumerate(stack_order):
            color = FRACTION_COLORS[key]
            label = stack_labels[key]
            nn_means = np.nan_to_num(
                _binned_mean(
                    frame[variable].to_numpy(),
                    frame[NN_FRACTIONS[key]].to_numpy(),
                    bins,
                ),
                nan=0.0,
            )
            classic_means = np.nan_to_num(
                _binned_mean(
                    frame[variable].to_numpy(),
                    classic_fractions[key],
                    bins,
                ),
                nan=0.0,
            )

            nn_lower = nn_cumulative.copy()
            nn_cumulative += nn_means
            axis.fill_between(
                bins,
                np.r_[nn_lower, nn_lower[-1]],
                np.r_[nn_cumulative, nn_cumulative[-1]],
                step="post",
                color=color,
                alpha=0.45,
                label=f"NN {label}",
            )

            classic_cumulative += classic_means
            classic_label = (
                "Classic QCD"
                if stack_index == 0
                else "Classic QCD+Wjets"
                if stack_index == 1
                else "Classic total"
            )
            axis.step(
                bins,
                np.r_[classic_cumulative, classic_cumulative[-1]],
                where="post",
                color=color,
                linestyle="--",
                linewidth=2.0,
                label=classic_label,
            )

        axis.set_xlim(bins[0], bins[-1])
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel(_get_label(labels_config, variable))
        axis.set_ylabel("Process fraction")
        axis.grid(True, which="major", alpha=0.25)
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.88),
            ncol=3,
            fontsize=9,
            frameon=False,
        )
        adjust_ylim_for_legend(axis, spacing=0.18)

        CMS_LABEL([axis])
        CMS_LUMI_TITLE([axis])
        CMS_CHANNEL_TITLE([axis])
        CMS_CATEGORY_TITLE([axis], title="fraction classifier")

        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(
                output_dir / f"training_fraction_{variable}.{extension}",
                dpi=200,
                bbox_inches="tight",
            )
        plt.close(fig)

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--masks-path", type=Path, default=MASKS_PATH)
    parser.add_argument("--training-var-path", type=Path, default=TRAINING_VAR_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--feature-store-path", type=Path, default=FEATURE_STORE_PATH)
    parser.add_argument("--feature-registry-path", type=Path, default=FEATURE_REGISTRY_PATH)
    parser.add_argument("--process-fractions-path", type=Path, default=DEFAULT_PROCESS_FRACTIONS_PATH)
    parser.add_argument("--plotting-config-path", type=Path, default=PLOTTING_CONFIG_PATH)
    parser.add_argument("--labels-config-path", type=Path, default=LABELS_CONFIG_PATH)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()

    frame = _load_frame_with_fraction_outputs(
        data_path=args.data_path,
        masks_path=args.masks_path,
        training_var_path=args.training_var_path,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
    )

    output_path = _store_fraction_output_frame(
        frame,
        feature_store_path=args.feature_store_path,
        feature_registry_path=args.feature_registry_path,
    )
    plots_dir = plot_fraction_comparisons(
        frame,
        output_dir=args.plots_dir,
        process_fractions_path=args.process_fractions_path,
        plotting_config_path=args.plotting_config_path,
        labels_config_path=args.labels_config_path,
    )
    print(f"Saved fraction NN outputs to {output_path}")
    print(f"Saved fraction comparison plots to {plots_dir}")


if __name__ == "__main__":
    main()
