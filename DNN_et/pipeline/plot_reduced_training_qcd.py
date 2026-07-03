import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from sklearn.model_selection import train_test_split

from classes import load_data, load_fold_combined_model, load_variables
from classes.enrichment_classifier import (
    QCD_SS_WEIGHT_DYNAMIC_DELTA,
    QCD_SS_WEIGHT_DYNAMIC_DELTA_LAST,
    QCD_SS_WEIGHT_DYNAMIC_MIN_QCD_YIELD,
    QCD_WEIGHT_BINNING,
    QCD_WEIGHT_N_BINS,
    build_qcd_weight_bins,
)
from groupings import GROUPING_NAMES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "dataframe_complete.feather"
DEFAULT_MASKS_PATH = PROJECT_ROOT / "configs" / "masks.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots" / "enrichment_qcd"
REDUCED_WEIGHT_STORE_DIR_QCD = (
    PROJECT_ROOT / "data" / "features" / "reduced_dataset" / "qcd"
)
QCD_FRACTION_MODEL_DIR = PROJECT_ROOT / "Enrichment_models" / "qcd_fraction"
QCD_FRACTION_WEIGHT_STORE_DIR = (
    PROJECT_ROOT / "data" / "features" / "qcd_fraction"
)
TRAINING_VARIABLES_PATH = (
    PROJECT_ROOT / "configs" / "training_variables_enrichment.yaml"
)
GROUPINGS = GROUPING_NAMES
TRAINING_SEED = 42

PROCESS_COMPONENTS = (
    ("wjets", "W+jets", "#e76300"),
    ("embedding", r"$\tau$ embedded", "#ffa90e"),
    ("diboson", "Diboson", "#94a4a2"),
    ("DYjets", r"DY+jets", "#3f90da"),
    ("ST", "Single top", "#717581"),
    ("ttbar", r"$t\bar{t}$", "#832db6"),
)
PROCESS_COMPONENTS_QCD_EXTRAPOLATION = (
    ("wjets", "W+jets", "#e76300"),
    ("diboson", "Diboson", "#94a4a2"),
    ("DYjets", r"DY+jets", "#3f90da"),
    ("ST", "Single top", "#717581"),
    ("ttbar", r"$t\bar{t}$", "#832db6"),
    ("diboson_T", r"Diboson true $\tau$", "#b6bfc7"),
    ("DYjets_T", r"DY+jets true $\tau$", "#92c5f9"),
    ("ST_T", r"Single top true $\tau$", "#a8abb3"),
    ("ttbar_T", r"$t\bar{t}$ true $\tau$", "#b37adb"),
)


def _safe_ratio(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )


def _equi_populated_bins(values, n_bins):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot build bins without finite NN outputs.")
    edges = np.quantile(
        values,
        np.linspace(0.0, 1.0, n_bins + 1),
    ).astype(values.dtype)
    edges = np.unique(edges)
    if len(edges) < 2:
        raise ValueError("Cannot build bins from a constant NN output.")
    return edges


def _histogram(values, bins, weights=None):
    valid = np.isfinite(values)
    if weights is not None:
        valid &= np.isfinite(weights)
        weights = weights[valid]
    return np.histogram(values[valid], bins=bins, weights=weights)[0]


def _torch_histogram(values, bins, weights=None):
    values_tensor = t.as_tensor(values, dtype=t.float32)
    bins_tensor = t.as_tensor(bins, dtype=t.float32)
    weight_tensor = None
    if weights is not None:
        weight_tensor = t.as_tensor(weights, dtype=t.float32)
    hist, _ = t.histogram(values_tensor, bins=bins_tensor, weight=weight_tensor)
    return hist.cpu().numpy()


def _load_column(region, column):
    values = np.asarray(region[column], dtype=float)
    if not np.isfinite(values).any():
        raise ValueError(
            f"{column} has no finite values. Re-run the task that produces "
            "this feature before plotting."
        )
    return values


def _qcd_fraction_region(df, process=None):
    return _configured_region(df, "DR_qcd_fractions", process=process)


def _configured_region(df, region_name, process=None):
    manager = df._manager
    if region_name in manager.regions:
        mask = manager.get_region_mask(df.events, region_name)
    else:
        mask = manager.get_mask(df.events, region_name)
        mask &= manager.get_mask(df.events, "preselection")
    if process is not None:
        mask &= manager.get_process_mask(df.events, process)
    return df.events.loc[mask]


def _load_group_metadata(model_dir, grouping):
    metadata_path = Path(model_dir) / grouping / "metadata.json"
    with open(metadata_path, "r") as handle:
        return json.load(handle)


def _group_mask(values, group):
    if len(group) == 1:
        return values == group[0]
    return (values >= group[0]) & (values <= group[1])


def _group_label(group):
    if len(group) == 1:
        return str(group[0])
    return f"{group[0]}-{group[1]}"


def _compute_fraction_histograms(
    df,
    inference_region,
    nn_output_name,
    qcd_weight_name,
    bins,
    process_components=PROCESS_COMPONENTS,
):
    """Compute histograms for QCD fractions using global equi-populated bins."""
    manager = df._manager
    inference_region = df.events.loc[inference_region.index]
    
    # QCD histograms
    qcd_region = inference_region.loc[inference_region["Label"] == 2]
    qcd_output = qcd_region[nn_output_name].to_numpy(dtype=np.float32)
    qcd_weight = qcd_region[qcd_weight_name].to_numpy(dtype=np.float32)
    weighted_qcd = _histogram(qcd_output, bins, qcd_weight)
    qcd_variance = _histogram(qcd_output, bins, qcd_weight ** 2)
    data_counts = _histogram(qcd_output, bins)
    
    # Process component histograms
    process_counts = {}
    process_variances = {}
    for process, _, _ in process_components:
        process_mask = manager.get_process_mask(inference_region, process)
        process_region = inference_region.loc[process_mask]
        process_output = process_region[nn_output_name].to_numpy(dtype=np.float32)
        process_weight = process_region["weight"].to_numpy(dtype=np.float32)
        process_counts[process] = _histogram(process_output, bins, process_weight)
        process_variances[process] = _histogram(process_output, bins, process_weight ** 2)
    
    return {
        "weighted_qcd": weighted_qcd,
        "qcd_variance": qcd_variance,
        "data_counts": data_counts,
        "process_counts": process_counts,
        "process_variances": process_variances,
    }


def _validate_qcd_fraction_region(region_df):
    if region_df.empty:
        raise ValueError("The QCD fractions region is empty.")
    if not (region_df["mt_1"] < 70).all():
        raise ValueError("The QCD fractions region contains mt_1 >= 70 events.")


def _predict_fold_output(model, region_df, training_variables, device):
    features = t.as_tensor(
        region_df[training_variables].to_numpy(dtype=np.float32),
        device=device,
    )
    parity = t.as_tensor(
        (region_df["event"] % 2).to_numpy(dtype=np.float32),
        device=device,
    ).unsqueeze(0)
    model_input = t.cat([parity, features.T], dim=0)
    return model(model_input).detach().cpu().numpy()


def _predict_weight_construction_output(
    even_model,
    odd_model,
    region_df,
    training_variables,
    device,
):
    """Reproduce the model assignment used when the QCD weights were built."""
    features = t.as_tensor(
        region_df[training_variables].to_numpy(dtype=np.float32),
        device=device,
    )
    odd_events = t.as_tensor(
        (region_df["event"] % 2 == 1).to_numpy(),
        dtype=t.bool,
        device=device,
    )
    even_output = even_model(features).squeeze()
    odd_output = odd_model(features).squeeze()
    return t.where(odd_events, even_output, odd_output).detach().cpu().numpy()


def _plot_fraction_closure(
    histograms,
    bins,
    grouping,
    output_dir,
    process_components=PROCESS_COMPONENTS,
    filename_prefix="qcd2",
    title_prefix="QCD fractions",
):
    """Plot QCD fractions closure validation."""
    weighted_qcd = histograms["weighted_qcd"]
    data_counts = histograms["data_counts"]
    background_counts = np.sum(
        [histograms["process_counts"][process] for process, _, _ in process_components],
        axis=0,
    )
    background_variance = np.sum(
        [histograms["process_variances"][process] for process, _, _ in process_components],
        axis=0,
    )
    expected_qcd = data_counts - background_counts
    expected_variance = data_counts + background_variance
    closure_delta = np.abs(weighted_qcd - expected_qcd)
    max_difference = float(np.max(closure_delta))

    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    ratio = _safe_ratio(weighted_qcd, expected_qcd)
    model_uncertainty = _safe_ratio(
        np.sqrt(expected_variance),
        np.abs(expected_qcd),
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 9),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05},
    )
    axes[0].errorbar(
        centers,
        weighted_qcd,
        xerr=0.5 * widths,
        yerr=np.sqrt(np.clip(weighted_qcd, 0.0, None)),
        fmt="o",
        color="black",
        label="Weighted data",
    )
    axes[0].bar(
        centers,
        expected_qcd,
        width=widths,
        color="#b9ac70",
        alpha=0.8,
        label="Data - simulated backgrounds",
    )
    axes[0].set_ylabel("Events")
    axes[0].set_title(
        f"{title_prefix} closure: {grouping} | max |Δ|={max_difference:.3g}"
    )
    axes[0].legend()

    axes[1].errorbar(
        centers,
        ratio,
        xerr=0.5 * widths,
        fmt="o",
        color="black",
    )
    axes[1].fill_between(
        centers,
        1.0 - model_uncertainty,
        1.0 + model_uncertainty,
        step="mid",
        color="gray",
        alpha=0.35,
        label="Subtraction stat. unc.",
    )
    axes[1].axhline(1.0, color="red", linestyle="--")
    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_ylabel("Data / Model")
    axes[1].set_xlabel("NN output")
    axes[1].legend()

    fig.savefig(
        output_dir / f"reduced_closure_{filename_prefix}_{grouping}.png",
        dpi=160,
    )
    fig.savefig(output_dir / f"reduced_closure_{filename_prefix}_{grouping}.pdf")
    plt.close(fig)


def _plot_fraction_training_composition(
    histograms,
    bins,
    grouping,
    output_dir,
    process_components=PROCESS_COMPONENTS,
    filename_prefix="qcd2",
    title_prefix="QCD fractions",
):
    """Plot QCD fractions training composition."""
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    component_counts = [
        histograms["process_counts"][process]
        for process, _, _ in process_components
    ]
    component_variances = [
        histograms["process_variances"][process]
        for process, _, _ in process_components
    ]
    component_counts.append(histograms["weighted_qcd"])
    component_variances.append(histograms["qcd_variance"])

    labels = [label for _, label, _ in process_components] + ["QCD multijet"]
    colors = [color for _, _, color in process_components] + ["#b9ac70"]
    simulation = np.sum(component_counts, axis=0)
    simulation_variance = np.sum(component_variances, axis=0)
    data_counts = histograms["data_counts"]
    max_difference = float(np.max(np.abs(simulation - data_counts)))
    ratio = _safe_ratio(data_counts, simulation)
    simulation_uncertainty = _safe_ratio(
        np.sqrt(simulation_variance),
        simulation,
    )
    fractions = [
        _safe_ratio(component, simulation)
        for component in component_counts
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 11),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1, 1), "hspace": 0.05},
    )
    bottom = np.zeros_like(simulation)
    for counts, label, color in zip(component_counts, labels, colors):
        axes[0].bar(
            centers,
            counts,
            width=widths,
            bottom=bottom,
            color=color,
            label=label,
        )
        bottom += counts

    axes[0].errorbar(
        centers,
        data_counts,
        xerr=0.5 * widths,
        yerr=np.sqrt(data_counts),
        fmt="o",
        color="black",
        label="Data",
    )
    axes[0].set_ylabel("Events")
    axes[0].set_title(
        f"{title_prefix} enrichment training: {grouping} | max |Δ|={max_difference:.3g}"
    )
    axes[0].legend(ncol=3, frameon=False)

    axes[1].errorbar(
        centers,
        ratio,
        xerr=0.5 * widths,
        fmt="o",
        color="black",
    )
    axes[1].fill_between(
        centers,
        1.0 - simulation_uncertainty,
        1.0 + simulation_uncertainty,
        step="mid",
        color="gray",
        alpha=0.35,
    )
    axes[1].axhline(1.0, color="red", linestyle="--")
    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_ylabel("Data / Sim.")

    bottom = np.zeros_like(simulation)
    for fraction, color in zip(fractions, colors):
        axes[2].bar(
            centers,
            fraction,
            width=widths,
            bottom=bottom,
            color=color,
        )
        bottom += np.nan_to_num(fraction)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("Proc. frac.")
    axes[2].set_xlabel("NN output")

    fig.savefig(
        output_dir / f"training_composition_{filename_prefix}_{grouping}.png",
        dpi=160,
    )
    fig.savefig(output_dir / f"training_composition_{filename_prefix}_{grouping}.pdf")
    plt.close(fig)


def _plot_reduced_closure(
    df,
    data_region,
    nn_output_name,
    reduced_weight_name,
    bins,
    grouping,
    output_dir,
):
    data_output = _load_column(data_region, nn_output_name)
    reduced_weight = _load_column(data_region, reduced_weight_name)

    reduced_counts = _histogram(
        data_output,
        bins,
        reduced_weight,
    )
    data_counts = _histogram(data_output, bins)
    background_counts = np.zeros_like(reduced_counts)
    background_variance = np.zeros_like(reduced_counts)
    for process, _, _ in PROCESS_COMPONENTS:
        process_region = getattr(df, process).DR_qcd
        process_output = _load_column(process_region, nn_output_name)
        process_weight = process_region.events["weight"].to_numpy(dtype=float)
        background_counts += _histogram(
            process_output,
            bins,
            process_weight,
        )
        background_variance += _histogram(
            process_output,
            bins,
            process_weight ** 2,
        )

    qcd_counts = data_counts - background_counts
    qcd_variance = data_counts + background_variance
    if not np.allclose(reduced_counts, qcd_counts, rtol=1e-5, atol=1e-5):
        max_difference = np.max(np.abs(reduced_counts - qcd_counts))
        raise RuntimeError(
            f"QCD/{grouping} plot closure failed; maximum bin "
            f"difference is {max_difference:.6g}. Re-run ReducedDataset."
        )

    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    ratio = _safe_ratio(reduced_counts, qcd_counts)
    model_uncertainty = _safe_ratio(
        np.sqrt(qcd_variance),
        np.abs(qcd_counts),
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 9),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05},
    )
    axes[0].errorbar(
        centers,
        reduced_counts,
        xerr=0.5 * widths,
        yerr=np.sqrt(np.clip(reduced_counts, 0.0, None)),
        fmt="o",
        color="black",
        label="Reduced data",
    )
    axes[0].bar(
        centers,
        qcd_counts,
        width=widths,
        color="#b9ac70",
        alpha=0.8,
        label="Data - simulated backgrounds",
    )
    axes[0].set_ylabel("Events")
    axes[0].set_title(f"QCD reduced-data closure: {grouping}")
    axes[0].legend()

    axes[1].errorbar(
        centers,
        ratio,
        xerr=0.5 * widths,
        fmt="o",
        color="black",
    )
    axes[1].fill_between(
        centers,
        1.0 - model_uncertainty,
        1.0 + model_uncertainty,
        step="mid",
        color="gray",
        alpha=0.35,
        label="Subtraction stat. unc.",
    )
    axes[1].axhline(1.0, color="red", linestyle="--")
    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_ylabel("Data / Model")
    axes[1].set_xlabel("NN output")
    axes[1].legend()

    fig.savefig(output_dir / f"reduced_closure_qcd_{grouping}.png", dpi=160)
    fig.savefig(output_dir / f"reduced_closure_qcd_{grouping}.pdf")
    plt.close(fig)


def _plot_training_composition(
    df,
    bins,
    grouping,
    output_dir,
):
    nn_output_name = f"nn_output_qcd_{grouping}"
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)

    component_counts = []
    component_variances = []
    for process, _, _ in PROCESS_COMPONENTS:
        process_region = getattr(df, process).DR_qcd
        process_output = _load_column(process_region, nn_output_name)
        process_weight = process_region.events["weight"].to_numpy(dtype=float)
        component_counts.append(_histogram(process_output, bins, process_weight))
        component_variances.append(
            _histogram(process_output, bins, process_weight ** 2)
        )

    data_region = df["data"].DR_qcd
    data_output = _load_column(data_region, nn_output_name)
    qcd_weight = _load_column(
        data_region,
        f"reduced_weight_qcd_{grouping}_nominal",
    )
    qcd_counts = _histogram(data_output, bins, qcd_weight)
    qcd_variance = _histogram(data_output, bins, qcd_weight ** 2)
    component_counts.append(qcd_counts)
    component_variances.append(qcd_variance)

    labels = [label for _, label, _ in PROCESS_COMPONENTS] + ["QCD multijet"]
    colors = [color for _, _, color in PROCESS_COMPONENTS] + ["#b9ac70"]
    simulation = np.sum(component_counts, axis=0)
    simulation_variance = np.sum(component_variances, axis=0)
    data_counts = _histogram(data_output, bins)
    if not np.allclose(simulation, data_counts, rtol=1e-5, atol=1e-5):
        max_difference = np.max(np.abs(simulation - data_counts))
        raise RuntimeError(
            f"QCD/{grouping} composition closure failed; maximum bin "
            f"difference is {max_difference:.6g}. Re-run ReducedDataset."
        )
    ratio = _safe_ratio(data_counts, simulation)
    simulation_uncertainty = _safe_ratio(
        np.sqrt(simulation_variance),
        simulation,
    )
    fractions = [
        _safe_ratio(component, simulation)
        for component in component_counts
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 11),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1, 1), "hspace": 0.05},
    )
    bottom = np.zeros_like(simulation)
    for counts, label, color in zip(component_counts, labels, colors):
        axes[0].bar(
            centers,
            counts,
            width=widths,
            bottom=bottom,
            color=color,
            label=label,
        )
        bottom += counts

    axes[0].errorbar(
        centers,
        data_counts,
        xerr=0.5 * widths,
        yerr=np.sqrt(data_counts),
        fmt="o",
        color="black",
        label="Data",
    )
    axes[0].set_ylabel("Events")
    axes[0].set_title(f"QCD enrichment training: {grouping}")
    axes[0].legend(ncol=3, frameon=False)

    axes[1].errorbar(
        centers,
        ratio,
        xerr=0.5 * widths,
        fmt="o",
        color="black",
    )
    axes[1].fill_between(
        centers,
        1.0 - simulation_uncertainty,
        1.0 + simulation_uncertainty,
        step="mid",
        color="gray",
        alpha=0.35,
    )
    axes[1].axhline(1.0, color="red", linestyle="--")
    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_ylabel("Data / Sim.")

    bottom = np.zeros_like(simulation)
    for fraction, color in zip(fractions, colors):
        axes[2].bar(
            centers,
            fraction,
            width=widths,
            bottom=bottom,
            color=color,
        )
        bottom += np.nan_to_num(fraction)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("Proc. frac.")
    axes[2].set_xlabel("NN output")

    fig.savefig(output_dir / f"training_composition_qcd_{grouping}.png", dpi=160)
    fig.savefig(output_dir / f"training_composition_qcd_{grouping}.pdf")
    plt.close(fig)


def create_qcd_training_plots(
    data_path=DEFAULT_DATA_PATH,
    masks_path=DEFAULT_MASKS_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    reduced_weight_store_dir=REDUCED_WEIGHT_STORE_DIR_QCD,
    n_bins=20,
):
    data_path = Path(data_path)
    masks_path = Path(masks_path)
    output_dir = Path(output_dir)
    reduced_weight_store_dir = Path(reduced_weight_store_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path, masks_path)
    data_region = df["data"].DR_qcd

    for grouping in GROUPINGS:
        df.load_feature_file(
            reduced_weight_store_dir
            / f"reduced_weight_{grouping}.feather"
        )
        nn_output_name = f"nn_output_qcd_{grouping}"
        reduced_weight_name = f"reduced_weight_qcd_{grouping}_nominal"
        data_values = _load_column(data_region, nn_output_name)
        bins = _equi_populated_bins(data_values, n_bins)

        _plot_training_composition(
            df,
            bins,
            grouping,
            output_dir,
        )
        _plot_reduced_closure(
            df,
            data_region,
            nn_output_name,
            reduced_weight_name,
            bins,
            grouping,
            output_dir,
        )

    print(f"Plots written to {output_dir}")


def create_qcd_fraction_training_plots(
    data_path=DEFAULT_DATA_PATH,
    masks_path=DEFAULT_MASKS_PATH,
    output_dir=PROJECT_ROOT / "plots" / "enrichment_qcd2",
    model_dir=QCD_FRACTION_MODEL_DIR,
    qcd_weight_store_dir=QCD_FRACTION_WEIGHT_STORE_DIR,
    training_variables_path=TRAINING_VARIABLES_PATH,
    n_bins=40,
    region_name="DR_qcd_fractions",
    model_output_prefix="qcd_fraction",
    feature_file_prefix="qcd_fraction_weights",
    weight_column_prefix="weight_qcd_fraction",
    plot_file_prefix="qcd2",
    title_prefix="QCD fractions",
    process_components=PROCESS_COMPONENTS,
):
    data_path = Path(data_path)
    masks_path = Path(masks_path)
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    qcd_weight_store_dir = Path(qcd_weight_store_dir)
    training_variables = load_variables(training_variables_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path, masks_path)
    inference_region = _configured_region(df, region_name).copy()
    _validate_qcd_fraction_region(inference_region)
    n_upper_mt = int(
        ((inference_region["mt_1"] >= 50) & (inference_region["mt_1"] < 70))
        .sum()
    )
    print(
        "QCD2 region audit: "
        f"{len(inference_region)} events, "
        f"{n_upper_mt} events with 50 <= mt_1 < 70"
    )
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    for grouping in GROUPINGS:
        df.load_feature_file(
            qcd_weight_store_dir
            / f"{feature_file_prefix}_{grouping}.feather"
        )
        model = load_fold_combined_model(
            even_model_path=model_dir / grouping / "fold_even",
            odd_model_path=model_dir / grouping / "fold_odd",
        ).to(device).eval()
        nn_output_name = f"nn_output_{model_output_prefix}_{grouping}"
        df.events.loc[inference_region.index, nn_output_name] = (
            _predict_weight_construction_output(
                model.even_model,
                model.odd_model,
                inference_region,
                training_variables,
                device,
            )
        )

        qcd_weight_name = f"{weight_column_prefix}_{grouping}"
        data_region = _configured_region(df, region_name, process="data")
        upper_mt_data = data_region[
            (data_region["mt_1"] >= 50)
            & (data_region["mt_1"] < 70)
        ]
        if not upper_mt_data.empty:
            upper_mt_weights = np.asarray(
                upper_mt_data[qcd_weight_name],
                dtype=float,
            )
            if not np.isfinite(upper_mt_weights).any():
                raise ValueError(
                    f"{qcd_weight_name} has no finite values for "
                    "50 <= mt_1 < 70. The feature file was likely produced "
                    "with the old DR_qcd mask and must be regenerated."
                )
        
        # Compute global equi-populated bins
        qcd_region = data_region.loc[data_region["Label"] == 2]
        qcd_output = qcd_region[nn_output_name].to_numpy(dtype=np.float32)
        bins = _equi_populated_bins(qcd_output, n_bins)
        
        # Compute histograms
        histograms = _compute_fraction_histograms(
            df,
            inference_region,
            nn_output_name,
            qcd_weight_name,
            bins,
            process_components=process_components,
        )
        
        # Generate plots
        _plot_fraction_training_composition(
            histograms,
            bins,
            grouping,
            output_dir,
            process_components=process_components,
            filename_prefix=plot_file_prefix,
            title_prefix=title_prefix,
        )
        _plot_fraction_closure(
            histograms,
            bins,
            grouping,
            output_dir,
            process_components=process_components,
            filename_prefix=plot_file_prefix,
            title_prefix=title_prefix,
        )

    print(f"Plots written to {output_dir}")


def create_qcd_extrapolation_training_plots(
    data_path=DEFAULT_DATA_PATH,
    masks_path=DEFAULT_MASKS_PATH,
    output_dir=PROJECT_ROOT / "plots" / "enrichment_qcd_extrapolation",
    model_dir=PROJECT_ROOT / "Enrichment_models" / "qcd_extrapolation",
    qcd_weight_store_dir=(
        PROJECT_ROOT / "data" / "features" / "qcd_extrapolation"
    ),
    training_variables_path=TRAINING_VARIABLES_PATH,
    n_bins=40,
):
    return create_qcd_fraction_training_plots(
        data_path=data_path,
        masks_path=masks_path,
        output_dir=output_dir,
        model_dir=model_dir,
        qcd_weight_store_dir=qcd_weight_store_dir,
        training_variables_path=training_variables_path,
        n_bins=n_bins,
        region_name="DR_qcd_extrapolation",
        model_output_prefix="qcd_extrapolation",
        feature_file_prefix="qcd_extrapolation_weights",
        weight_column_prefix="weight_qcd_extrapolation",
        plot_file_prefix="qcd_extrapolation",
        title_prefix="QCD extrapolation",
        process_components=PROCESS_COMPONENTS_QCD_EXTRAPOLATION,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--masks", type=Path, default=DEFAULT_MASKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bins", type=int, default=20)
    args = parser.parse_args()

    create_qcd_training_plots(
        data_path=args.data,
        masks_path=args.masks,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
    )


if __name__ == "__main__":
    main()
