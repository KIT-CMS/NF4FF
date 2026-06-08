import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from classes import load_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "dataframe_complete.feather"
DEFAULT_MASKS_PATH = PROJECT_ROOT / "configs" / "masks.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots" / "enrichment_wjets"
QCD_WEIGHT_STORE_DIR_WJETS = PROJECT_ROOT / "data" / "features" / "wjets"
GROUPINGS = ("tau_decaymode_2", "njets")

PROCESS_COMPONENTS = (
    ("wjets", "W+jets", "#e76300"),
    ("embedding", r"$\tau$ embedded", "#ffa90e"),
    ("diboson", "Diboson", "#94a4a2"),
    ("DYjets", r"DY+jets", "#3f90da"),
    ("ST", "Single top", "#717581"),
    ("ttbar", r"$t\bar{t}$", "#832db6"),
)


def _safe_ratio(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )


def _equi_populated_bins(values, n_bins):
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
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


def _load_column(region, column):
    values = np.asarray(region[column], dtype=float)
    if not np.isfinite(values).any():
        raise ValueError(
            f"{column} has no finite values. Re-run the ReducedDataset task "
            "after updating the workflow."
        )
    return values


def _plot_reduced_closure(
    data_region,
    wjets_region,
    nn_output_name,
    reduced_weight_name,
    bins,
    grouping,
    output_dir,
):
    data_output = _load_column(data_region, nn_output_name)
    reduced_weight = _load_column(data_region, reduced_weight_name)
    wjets_output = _load_column(wjets_region, nn_output_name)
    wjets_weight = wjets_region.events["weight"].to_numpy(dtype=float)

    reduced_counts = _histogram(
        data_output,
        bins,
        reduced_weight,
    )
    wjets_counts = _histogram(
        wjets_output,
        bins,
        wjets_weight,
    )
    wjets_variance = _histogram(
        wjets_output,
        bins,
        wjets_weight ** 2,
    )

    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    ratio = _safe_ratio(reduced_counts, wjets_counts)
    model_uncertainty = _safe_ratio(np.sqrt(wjets_variance), wjets_counts)

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
        wjets_counts,
        width=widths,
        color="#e76300",
        alpha=0.8,
        label="W+jets simulation",
    )
    axes[0].set_ylabel("Events")
    axes[0].set_title(f"W+jets reduced-data closure: {grouping}")
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
        label="Simulation stat. unc.",
    )
    axes[1].axhline(1.0, color="red", linestyle="--")
    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_ylabel("Data / Model")
    axes[1].set_xlabel("NN output")
    axes[1].legend()

    fig.savefig(output_dir / f"reduced_closure_wjets_{grouping}.png", dpi=160)
    fig.savefig(output_dir / f"reduced_closure_wjets_{grouping}.pdf")
    plt.close(fig)


def _plot_training_composition(
    df,
    bins,
    grouping,
    output_dir,
):
    nn_output_name = f"nn_output_wjets_{grouping}"
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)

    component_counts = []
    component_variances = []
    for process, _, _ in PROCESS_COMPONENTS:
        process_region = getattr(df, process).DR_wjets
        process_output = _load_column(process_region, nn_output_name)
        process_weight = process_region.events["weight"].to_numpy(dtype=float)
        component_counts.append(_histogram(process_output, bins, process_weight))
        component_variances.append(
            _histogram(process_output, bins, process_weight ** 2)
        )

    data_region = df["data"].DR_wjets
    qcd_region = df["data"].DR_wjets_without_signs
    data_output = _load_column(data_region, nn_output_name)
    qcd_output = _load_column(qcd_region, nn_output_name)
    qcd_weight = _load_column(qcd_region, f"qcd_weight_{grouping}")
    qcd_counts = _histogram(qcd_output, bins, qcd_weight)
    qcd_variance = _histogram(qcd_output, bins, qcd_weight ** 2)
    component_counts.append(qcd_counts)
    component_variances.append(qcd_variance)

    labels = [label for _, label, _ in PROCESS_COMPONENTS] + ["QCD multijet"]
    colors = [color for _, _, color in PROCESS_COMPONENTS] + ["#b9ac70"]
    simulation = np.sum(component_counts, axis=0)
    simulation_variance = np.sum(component_variances, axis=0)
    data_counts = _histogram(data_output, bins)
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
    axes[0].set_title(f"W+jets enrichment training: {grouping}")
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

    fig.savefig(output_dir / f"training_composition_wjets_{grouping}.png", dpi=160)
    fig.savefig(output_dir / f"training_composition_wjets_{grouping}.pdf")
    plt.close(fig)


def create_wjets_training_plots(
    data_path=DEFAULT_DATA_PATH,
    masks_path=DEFAULT_MASKS_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    qcd_weight_store_dir=QCD_WEIGHT_STORE_DIR_WJETS,
    n_bins=20,
):
    data_path = Path(data_path)
    masks_path = Path(masks_path)
    output_dir = Path(output_dir)
    qcd_weight_store_dir = Path(qcd_weight_store_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path, masks_path)
    data_region = df["data"].DR_wjets
    wjets_region = df.wjets.DR_wjets

    for grouping in GROUPINGS:
        df.load_feature_file(
            qcd_weight_store_dir / f"qcd_weights_{grouping}.feather"
        )
        nn_output_name = f"nn_output_wjets_{grouping}"
        reduced_weight_name = f"reduced_weight_wjets_{grouping}_nominal"
        data_values = _load_column(data_region, nn_output_name)
        bins = _equi_populated_bins(data_values, n_bins)

        _plot_training_composition(
            df,
            bins,
            grouping,
            output_dir,
        )
        _plot_reduced_closure(
            data_region,
            wjets_region,
            nn_output_name,
            reduced_weight_name,
            bins,
            grouping,
            output_dir,
        )

    print(f"Plots written to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--masks", type=Path, default=DEFAULT_MASKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bins", type=int, default=20)
    args = parser.parse_args()

    create_wjets_training_plots(
        data_path=args.data,
        masks_path=args.masks,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
    )


if __name__ == "__main__":
    main()
