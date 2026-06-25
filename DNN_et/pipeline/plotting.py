import math
import json
import logging
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from classes import (
    FF_closure_in_DR_qcd,
    FF_closure_in_DR_ttbar,
    FF_closure_in_DR_wjets,
    load_data,
    plot_fake_factors_grouped,
    plot_fake_factors_grouped_range,
    plot_fake_factors_in_dr_grouped,
    plot_fake_factors_in_dr_grouped_range,
)
from groupings import GROUPING_NAMES, grouping_suffix


logger = logging.getLogger(__name__)


def _install_feature_aliases(df, prefix: str, feature_suffix: str) -> None:
    if not feature_suffix:
        return

    matching_columns = [
        column
        for column in df.events.columns
        if column.startswith(prefix) and column.endswith(feature_suffix)
    ]
    if not matching_columns:
        raise KeyError(
            f"No {prefix} features found with suffix '{feature_suffix}'."
        )

    for source in matching_columns:
        alias = source[:-len(feature_suffix)]
        df.events[alias] = df.events[source]


def _grouping_title(grouping):
    if grouping == "tau_decaymode_2_alt":
        return "tau_decaymode_2"
    return grouping


PLOT_SUBSETS = (
    ("inclusive", "inclusive", None),
    # ("njets_eq_0", r"$N_{\mathrm{jets}} = 0$", ("njets", "eq", 0)),
    # ("njets_eq_1", r"$N_{\mathrm{jets}} = 1$", ("njets", "eq", 1)),
    # ("njets_ge_2", r"$N_{\mathrm{jets}} \geq 2$", ("njets", "ge", 2)),
    # (
    #     "tau_decaymode_2_eq_0",
    #     r"$\tau_{\mathrm{DM}} = 0$",
    #     ("tau_decaymode_2", "eq", 0),
    # ),
    # (
    #     "tau_decaymode_2_eq_1",
    #     r"$\tau_{\mathrm{DM}} = 1$",
    #     ("tau_decaymode_2", "eq", 1),
    # ),
    # (
    #     "tau_decaymode_2_eq_10",
    #     r"$\tau_{\mathrm{DM}} = 10$",
    #     ("tau_decaymode_2", "eq", 10),
    # ),
    # (
    #     "tau_decaymode_2_eq_11",
    #     r"$\tau_{\mathrm{DM}} = 11$",
    #     ("tau_decaymode_2", "eq", 11),
    # ),
)

CLOSURE_ONLY_SUBSETS = (

    ("njets_eq_0", r"$N_{\mathrm{jets}} = 0$", ("njets", "eq", 0)),
    ("njets_eq_1", r"$N_{\mathrm{jets}} = 1$", ("njets", "eq", 1)),
    ("njets_ge_2", r"$N_{\mathrm{jets}} \geq 2$", ("njets", "ge", 2)),
    (
        "tau_decaymode_2_eq_0",
        r"$\tau_{\mathrm{DM}} = 0$",
        ("tau_decaymode_2", "eq", 0),
    ),
    (
        "tau_decaymode_2_eq_1",
        r"$\tau_{\mathrm{DM}} = 1$",
        ("tau_decaymode_2", "eq", 1),
    ),
    (
        "tau_decaymode_2_eq_10",
        r"$\tau_{\mathrm{DM}} = 10$",
        ("tau_decaymode_2", "eq", 10),
    ),
    (
        "tau_decaymode_2_eq_11",
        r"$\tau_{\mathrm{DM}} = 11$",
        ("tau_decaymode_2", "eq", 11),
    ),
    (
        "tau_decaymode_2_in_0_1",
        r"$\tau_{\mathrm{DM}} \in \{0, 1\}$",
        ("tau_decaymode_2", "isin", (0, 1)),
    ),
    (
        "tau_decaymode_2_in_10_11",
        r"$\tau_{\mathrm{DM}} \in \{10, 11\}$",
        ("tau_decaymode_2", "isin", (10, 11)),
    ),
)

HIGH_FF_CLOSURE_GROUPINGS = {
    "inclusive": GROUPING_NAMES,
    "njets_eq_0": ("njets",),
    "njets_eq_1": ("njets",),
    "njets_ge_2": ("njets",),
    "tau_decaymode_2_eq_0": ("tau_decaymode_2",),
    "tau_decaymode_2_eq_1": ("tau_decaymode_2",),
    "tau_decaymode_2_eq_10": ("tau_decaymode_2",),
    "tau_decaymode_2_eq_11": ("tau_decaymode_2",),
    "tau_decaymode_2_in_0_1": ("tau_decaymode_2_alt",),
    "tau_decaymode_2_in_10_11": ("tau_decaymode_2_alt",),
}


def _opposite_distribution_grouping(grouping: str) -> str:
    if grouping == "njets":
        return "tau_decaymode_2"
    if grouping in ("tau_decaymode", "tau_decaymode_2", "tau_decaymode_2_alt"):
        return "njets"
    raise ValueError(f"Unsupported grouping: {grouping}")


def _require_grouped_fake_factor_features(df) -> None:
    required_features = (
        "ff_dnn",
        "ff_dnn_tau_decaymode_2_alt",
        "ff_dnn_njets",
        "ff_dnn_wjets",
        "ff_dnn_qcd",
        "ff_dnn_ttbar",
        "ff_dnn_wjets_tau_decaymode_2_alt",
        "ff_dnn_qcd_tau_decaymode_2_alt",
        "ff_dnn_ttbar_tau_decaymode_2_alt",
        "ff_dnn_wjets_njets",
        "ff_dnn_qcd_njets",
        "ff_dnn_ttbar_njets",
    )
    missing = [
        feature
        for feature in required_features
        if feature not in df.events.columns
    ]
    if missing:
        raise KeyError(
            f"Fake-factor feature file is missing columns: {missing}"
        )


def _plot_subset(df, selection):
    if selection is None:
        return df

    column, operation, value = selection
    if column not in df.events.columns:
        raise KeyError(f"Missing plot-subset column: {column}")
    if operation == "eq":
        mask = df.events[column] == value
    elif operation == "ge":
        mask = df.events[column] >= value
    elif operation == "isin":
        mask = df.events[column].isin(value)
    else:
        raise ValueError(f"Unsupported plot-subset operation: {operation}")
    return df.subset(mask)


def CMS_CHANNEL_TITLE(ax, *args, **kwargs):
    ax[0].set_title(
        r"$e\tau_h$",
        fontsize=20,
        loc="left",
        fontproperties="Tex Gyre Heros"
    )

def CMS_CATEGORY_TITLE(ax, title="tau_DM: inclusive", *args, **kwargs):
    ax[0].set_title(
        title,
        fontsize=10,
        loc="center",
        fontproperties="Tex Gyre Heros"
    )

def CMS_LUMI_TITLE(ax, *args, **kwargs):
    ax[0].set_title(
        "59.8 $fb^{-1}$ (2018, 13 TeV)",
        fontsize=20,
        loc="right",
        fontproperties="Tex Gyre Heros"
    )

def CMS_LABEL(ax, *args, **kwargs):
    ax[0].text(
        0.025, 0.95,
        "Private work (CMS data/simulation)",
        fontsize=15,
        verticalalignment='top',
        fontproperties="Tex Gyre Heros:italic",
        bbox=dict(facecolor="white", alpha=0, edgecolor="white", boxstyle="round,pad=0.5"),
        transform=ax[0].transAxes
    )


def reorder_for_rowwise_legend(handles, labels, ncol, reverse=False):
    if reverse:
        handles = handles[::-1]
        labels = labels[::-1]

    n = len(handles)
    nrows = math.ceil(n / ncol)

    new_handles, new_labels = [], []

    for col in range(ncol):
        for row in range(nrows):
            idx = row * ncol + col
            if idx < n:
                new_handles.append(handles[idx])
                new_labels.append(labels[idx])

    return new_handles, new_labels

def adjust_ylim_for_legend(ax=None, spacing=0.05):
    if ax is None:
        ax = plt.gca()

    fig = ax.figure
    fig.canvas.draw()

    if (leg := ax.get_legend()) is None:
        return

    bbox_leg, bbox_ax = leg.get_window_extent(), ax.get_window_extent()

    legend_height_ratio = bbox_leg.height / bbox_ax.height

    ymin, ymax = ax.get_ylim()
    scale = ax.get_yscale()

    if (available_fraction := 1.0 - legend_height_ratio - spacing) <= 0.1:
        available_fraction = 0.1

    if scale == "linear":
        data_max_y = ax.dataLim.y1
        data_range = data_max_y - ymin
        new_range = data_range / available_fraction
        new_ymax = ymin + new_range
        ax.set_ylim(ymin, new_ymax)

    elif scale == "log":
        log_ymin = np.log10(ymin)
        log_data_max = np.log10(ax.dataLim.y1)
        log_range = log_data_max - log_ymin
        new_log_range = log_range / available_fraction
        new_log_ymax = log_ymin + new_log_range

        new_log_ymax = np.ceil(new_log_ymax)

        new_ymax = 10 ** new_log_ymax
        ax.set_ylim(ymin, new_ymax)

def estimate_jet_fakes(
	df,
	bins,
	var,
	ff_var,
):
	counts = {}
	variance = {}
	list_processes = ['data', 'diboson', 'DYjets', 'ST', 'embedding', 'ttbar_L']
	for proc in list_processes:	
		counts[proc], _ = np.histogram(df[proc].AR[var], weights = df[proc].AR.weight * df[proc].AR[ff_var], bins = bins)
		variance[proc], _ = np.histogram(df[proc].AR[var], weights = (df[proc].AR.weight * df[proc].AR[ff_var])**2, bins = bins)

	jet_fakes = counts['data'] - counts['diboson'] - counts['DYjets'] - counts['ST'] - counts['embedding'] - counts['ttbar_L']
	var_jet_fakes = variance['data'] + variance['diboson'] + variance['DYjets'] + variance['ST'] + variance['embedding'] + variance['ttbar_L']

	return jet_fakes, var_jet_fakes

def weighted_histogram(values, weights, bins):
    counts, edges = np.histogram(values, weights=weights, bins=bins)
    variances, _ = np.histogram(values, weights=weights**2, bins=bins)
    return counts, variances, edges

def draw_stacked_stepfill(ax, bin_edges, components: list[tuple[np.ndarray, str, str]]) -> np.ndarray:
    cumulative = np.zeros(len(bin_edges) - 1, dtype=float)
    final_top = cumulative.copy()

    for counts, color, label in components:
        next_cumulative = cumulative + counts
        ax.fill_between(
            bin_edges,
            np.r_[cumulative, cumulative[-1]],
            np.r_[next_cumulative, next_cumulative[-1]],
            step='post',
            color=color,
            linewidth=0,
            label=label,
        )
        ax.stairs(next_cumulative, bin_edges, color='black', linewidth=1.0)
        cumulative = next_cumulative
        final_top = next_cumulative

    return final_top

def plot_closure(
    df,
    var: str,
    bins: np.ndarray,
    label: str,
    grouping = 'tau_decaymode',
    corr_emb_ff = 1.0,
    plot_classic_ff_comp = True,
    plot_corr_hline = True,
):
    ff_dnn = f"ff_dnn{grouping_suffix(grouping)}"

    histograms = {}

    list_processes = [
        'data',
        'diboson',
        'DYjets',
        'ST',
        'ttbar_L',
        'embedding',
    ]


    for proc in list_processes:

        proc_counts, proc_variance, bin_edges = weighted_histogram(
            values=df[proc].SR[var],
            weights=df[proc].SR.weight,
            bins=bins,
        )

        histograms[proc] = {
            'counts': proc_counts,
            'variance': proc_variance,
        }


    classic_ff_variable = 'ff_classic' if plot_classic_ff_comp else ff_dnn
    jet_fakes_classic, var_jet_fakes_classic = estimate_jet_fakes(
        df,
        bins,
        var,
        classic_ff_variable,
    )

    jet_fakes_dnn, var_jet_fakes_dnn = estimate_jet_fakes(
        df,
        bins,
        var,
        ff_dnn,
    )

    histograms['jet_fakes_classic'] = {
        'counts': jet_fakes_classic,
        'variance': var_jet_fakes_classic,
    }

    histograms['jet_fakes_dnn'] = {
        'counts': jet_fakes_dnn,
        'variance': var_jet_fakes_dnn,
    }

    background_dnn = (
        histograms['diboson']['counts']
        + histograms['DYjets']['counts']
        + histograms['ST']['counts']
        + histograms['ttbar_L']['counts']
        + histograms['embedding']['counts']
        + histograms['jet_fakes_dnn']['counts']
    )

    background_classic = (
        histograms['diboson']['counts']
        + histograms['DYjets']['counts']
        + histograms['ST']['counts']
        + histograms['ttbar_L']['counts']
        + histograms['embedding']['counts']
        + histograms['jet_fakes_classic']['counts']
    )

    variance_background_dnn = (
        histograms['diboson']['variance']
        + histograms['DYjets']['variance']
        + histograms['ST']['variance']
        + histograms['ttbar_L']['variance']
        + histograms['embedding']['variance']
        + histograms['jet_fakes_dnn']['variance']
    )

    variance_background_classic = (
        histograms['diboson']['variance']
        + histograms['DYjets']['variance']
        + histograms['ST']['variance']
        + histograms['ttbar_L']['variance']
        + histograms['embedding']['variance']
        + histograms['jet_fakes_classic']['variance']
    )

    histograms['background_dnn'] = {
        'counts': background_dnn,
        'variance': variance_background_dnn,
    }

    histograms['background_classic'] = {
        'counts': background_classic,
        'variance': variance_background_classic,
    }


    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    err_bin = 0.5 * bin_widths


    err_data = np.sqrt(histograms['data']['variance'])


    err_stat_dnn = np.sqrt(
        histograms['background_dnn']['variance']
    )

    err_stat_classic = np.sqrt(
        histograms['background_classic']['variance']
    )


    err_stat_rel_dnn = np.divide(
        err_stat_dnn,
        histograms['background_dnn']['counts'],
        out=np.zeros_like(err_stat_dnn),
        where=histograms['background_dnn']['counts'] > 0,
    )

    err_stat_rel_classic = np.divide(
        err_stat_classic,
        histograms['background_classic']['counts'],
        out=np.zeros_like(err_stat_classic),
        where=histograms['background_classic']['counts'] > 0,
    )
    if plot_classic_ff_comp == True:
        fig, ax = plt.subplots(
            4,
            1,
            figsize=(9, 9),
            sharex=True,
            gridspec_kw={
                'height_ratios': [4, 1, 0.2, 1],
                'hspace': 0.05,
            },
            constrained_layout=True,
        )
    else:
        fig, ax = plt.subplots(
            2,
            1,
            figsize=(9, 7),
            sharex=True,
            gridspec_kw={
                'height_ratios': [3, 1],
                'hspace': 0.05,
            },
            constrained_layout=True,
        )    

    stack_components = [
        (histograms['diboson']['counts'], "#94a4a2", 'Diboson'),
        (histograms['ttbar_L']['counts'], '#832db6', r'$t\bar{t} \to \tau$'),
        (histograms['ST']['counts'], "#717581", r"Single t"),
        (histograms['DYjets']['counts'], '#3f90da', r'$Z \to \ell \ell$'),
        (histograms['jet_fakes_dnn']['counts'], "#a96b59", r'Jet $\rightarrow \tau_h$'),
        (histograms['embedding']['counts'], '#ffa90e', r'$\tau$ embedded'),
    ]

    counts_stack_total = draw_stacked_stepfill(
        ax[0],
        bin_edges,
        stack_components,
    )


    ax[0].stairs(
        counts_stack_total,
        bin_edges,
        color='black',
        linewidth=0.7,
    )

    ax[0].errorbar(
        bin_centers,
        histograms['data']['counts'],
        yerr=err_data,
        xerr=err_bin,
        fmt='o',
        color='black',
        label='Data',
        markersize=6,
        elinewidth=1.2,
        capsize=0,
    )

    ax[0].set_ylabel("Events")
    handles, labels = ax[0].get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    handles, labels = reorder_for_rowwise_legend(handles, labels, ncol=4)
    ax[0].legend(handles, labels, title=' ', title_fontsize=20, loc='upper left', ncol=4, frameon=False)
    adjust_ylim_for_legend(ax[0])
    ax[0].tick_params(direction='in', top=True, right=True)

    CMS_LABEL(ax)
    #CMS_CATEGORY_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_CHANNEL_TITLE(ax)

    ratio_dnn = np.divide(
        histograms['data']['counts'],
        histograms['background_dnn']['counts'],
        out=np.zeros_like(histograms['data']['counts'], dtype=float),
        where=histograms['background_dnn']['counts'] > 0,
    )

    ratio_err_dnn = np.divide(
        err_data,
        histograms['background_dnn']['counts'],
        out=np.zeros_like(err_data),
        where=histograms['background_dnn']['counts'] > 0,
    )

    ax[1].errorbar(
        bin_centers,
        ratio_dnn,
        xerr=err_bin,
        yerr=ratio_err_dnn,
        fmt='o',
        color='black',
        markersize=6,
        label=r'DNN $F_\mathrm{F}$',
    )

    ax[1].fill_between(
        bin_centers,
        1 - err_stat_rel_dnn,
        1 + err_stat_rel_dnn,
        color='gray',
        alpha=0.3,
        step='mid',
        label='Stat. Unc.',
    )
    if plot_corr_hline == True:
        ax[1].axhline(1/corr_emb_ff, color='blue', linestyle='--', linewidth=1.5)
    ax[1].axhline(1, color='red', linestyle='--', linewidth=1.5)
    
    ax[1].set_ylabel("Data / Model")
    ax[1].set_ylim([0.75, 1.25])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=2, frameon=False)

    ratio_classic = np.divide(
        histograms['data']['counts'],
        histograms['background_classic']['counts'],
        out=np.zeros_like(histograms['data']['counts'], dtype=float),
        where=histograms['background_classic']['counts'] > 0,
    )

    ratio_err_classic = np.divide(
        err_data,
        histograms['background_classic']['counts'],
        out=np.zeros_like(err_data),
        where=histograms['background_classic']['counts'] > 0,
    )

    if plot_classic_ff_comp == True:

        ax[2].axis('off')

        ax[3].errorbar(
            bin_centers,
            ratio_classic,
            xerr=err_bin,
            yerr=ratio_err_classic,
            fmt='o',
            color='black',
            markersize=6,
            label=r'Classic $F_\mathrm{F}$',
        )

        ax[3].fill_between(
            bin_centers,
            1 - err_stat_rel_classic,
            1 + err_stat_rel_classic,
            color='gray',
            alpha=0.3,
            step='mid',
            label='Stat. Unc.',
        )
        ax[3].axhline(1/corr_emb_ff, color='blue', linestyle='--', linewidth=1.5)
        ax[3].axhline(1, color='red', linestyle='--', linewidth=1.5)
        ax[3].set_ylabel("Data / Model")
        ax[3].set_ylim([0.75, 1.25])
        ax[3].grid(True, linestyle=':', alpha=0.7)
        ax[3].tick_params(direction='in', top=True, right=True)
        ax[3].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=2, frameon=False)
        ax[3].set_xlabel(label)
    else:
        ax[1].set_xlabel(label)

    return fig, ax, histograms


def _read_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _read_channel_labels(path: Union[str, Path], channel: str) -> dict:
    labels = {}
    current_channel = None
    with open(path, "r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            indent = len(line) - len(line.lstrip(" "))
            if not stripped or stripped.startswith("#"):
                continue
            if (
                stripped.endswith(":")
                and ":" not in stripped[:-1]
                and indent <= 1
            ):
                current_channel = stripped[:-1]
                continue
            if current_channel != channel or indent < 4 or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            labels[key] = value.strip().strip('"').strip("'")
    return labels


def _plot_bins(bin_spec) -> np.ndarray:
    if isinstance(bin_spec, (list, tuple)) and len(bin_spec) == 3:
        start, stop, count = bin_spec
        return np.linspace(float(start), float(stop), int(count))
    return np.asarray(bin_spec, dtype=float)


def _save_figure(fig, output_base: Path) -> list[str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("png", "pdf"):
        output_path = output_base.with_suffix(f".{extension}")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        outputs.append(str(output_path))
    plt.close(fig)
    return outputs


def _high_ff_model_plot(
    df,
    *,
    variable: str,
    bins: np.ndarray,
    label: str,
    grouping: str,
    category_title: str,
):
    ff_column = f"ff_dnn{grouping_suffix(grouping)}"
    high_ff_df = df.subset(df.events[ff_column] > 1.0)
    components = []
    total_variance = np.zeros(len(bins) - 1, dtype=float)
    component_specs = (
        ("diboson", "#94a4a2", "Diboson"),
        ("ttbar_L", "#832db6", r"$t\bar{t} \to \tau$"),
        ("ST", "#717581", "Single t"),
        ("DYjets", "#3f90da", r"$Z \to \ell\ell$"),
    )
    for process, color, process_label in component_specs:
        counts, variance, _ = weighted_histogram(
            high_ff_df[process].SR[variable],
            high_ff_df[process].SR.weight,
            bins,
        )
        components.append((counts, color, process_label))
        total_variance += variance

    jet_fakes, jet_fakes_variance = estimate_jet_fakes(
        high_ff_df,
        bins,
        variable,
        ff_column,
    )
    components.append((jet_fakes, "#a96b59", r"Jet $\rightarrow \tau_h$"))
    total_variance += jet_fakes_variance

    embedding_counts, embedding_variance, _ = weighted_histogram(
        high_ff_df.embedding.SR[variable],
        high_ff_df.embedding.SR.weight,
        bins,
    )
    components.append((embedding_counts, "#ffa90e", r"$\tau$ embedded"))
    total_variance += embedding_variance

    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    total = draw_stacked_stepfill(axis, bins, components)
    uncertainty = np.sqrt(total_variance)
    axis.fill_between(
        bins,
        np.r_[total - uncertainty, (total - uncertainty)[-1]],
        np.r_[total + uncertainty, (total + uncertainty)[-1]],
        step="post",
        color="black",
        alpha=0.18,
        linewidth=0,
        label="Stat. unc.",
    )
    axis.set_xlabel(label)
    axis.set_ylabel("Predicted events")
    axis.tick_params(direction="in", top=True, right=True)
    axis.legend(frameon=False, ncol=2)
    adjust_ylim_for_legend(axis)
    CMS_LABEL([axis])
    CMS_LUMI_TITLE([axis])
    CMS_CHANNEL_TITLE([axis])
    CMS_CATEGORY_TITLE(
        [axis],
        f"{category_title}, {_grouping_title(grouping)}, "
        r"$F_\mathrm{F} > 1$",
    )
    return fig


def _high_ff_process_plot(
    df,
    *,
    process: str,
    variable: str,
    bins: np.ndarray,
    label: str,
    grouping: str,
    category_title: str,
):
    ff_column = f"ff_dnn_{process}{grouping_suffix(grouping)}"
    high_ff_df = df.subset(df.events[ff_column] > 1.0)
    process_view = getattr(high_ff_df.data, f"AR_like_{process}")
    weight_column = (
        f"reduced_weight_{process}_{grouping}_nominal"
        if process in ("wjets", "qcd")
        else "weight"
    )
    weights = process_view[weight_column] * process_view[ff_column]
    counts, variance, bin_edges = weighted_histogram(
        process_view[variable],
        weights,
        bins,
    )
    uncertainty = np.sqrt(variance)

    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    axis.stairs(
        counts,
        bin_edges,
        color="#a96b59",
        linewidth=2,
        label=rf"$F_\mathrm{{F}}^{{{process}}}\cdot$ AR-like prediction",
    )
    axis.fill_between(
        bin_edges,
        np.r_[counts - uncertainty, (counts - uncertainty)[-1]],
        np.r_[counts + uncertainty, (counts + uncertainty)[-1]],
        step="post",
        color="#a96b59",
        alpha=0.25,
        linewidth=0,
        label="Stat. unc.",
    )
    axis.set_xlabel(label)
    axis.set_ylabel("Predicted events")
    axis.tick_params(direction="in", top=True, right=True)
    axis.legend(frameon=False)
    adjust_ylim_for_legend(axis)
    CMS_LABEL([axis])
    CMS_LUMI_TITLE([axis])
    CMS_CHANNEL_TITLE([axis])
    CMS_CATEGORY_TITLE(
        [axis],
        f"{category_title}, {_grouping_title(grouping)}, "
        rf"$F_\mathrm{{F}}^{{{process}}} > 1$",
    )
    return fig


def create_high_ff_closure_plots(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    feature_path: Union[str, Path],
    reduced_weight_paths: Iterable[Union[str, Path]],
    plotting_config_path: Union[str, Path],
    labels_path: Union[str, Path],
    output_dir: Union[str, Path],
    manifest_path: Union[str, Path, None] = None,
    variable_set: str = "variables_set_small",
    channel: str = "et",
    feature_suffix: str = "",
) -> list[str]:
    """Plot model-only closure predictions for events with a DNN FF above one."""
    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    plotting_config = _read_yaml(plotting_config_path)
    labels = _read_channel_labels(labels_path, channel)
    variables = plotting_config.get(variable_set, [])
    binning = plotting_config.get("bins_by_variable", {})
    if not variables:
        raise ValueError(f"No plotting variables configured in {variable_set}.")

    df = load_data(data_path, masks_path)
    for path in (feature_path, *reduced_weight_paths):
        df.load_feature_file(path)
    _install_feature_aliases(df, "ff_dnn", feature_suffix)

    outputs = []
    subsets = (*PLOT_SUBSETS, *CLOSURE_ONLY_SUBSETS)
    for subset_name, subset_title, selection in subsets:
        subset_df = _plot_subset(df, selection)
        for grouping in HIGH_FF_CLOSURE_GROUPINGS[subset_name]:
            for variable in variables:
                if variable not in binning:
                    raise KeyError(f"No binning configured for {variable}.")
                bins = _plot_bins(binning[variable])
                label = labels.get(variable, variable)

                fig = _high_ff_model_plot(
                    subset_df,
                    variable=variable,
                    bins=bins,
                    label=label,
                    grouping=grouping,
                    category_title=subset_title,
                )
                outputs.extend(_save_figure(
                    fig,
                    output_dir
                    / subset_name
                    / "closure"
                    / grouping
                    / f"closure_{variable}",
                ))

                for process in ("wjets", "qcd", "ttbar"):
                    fig = _high_ff_process_plot(
                        subset_df,
                        process=process,
                        variable=variable,
                        bins=bins,
                        label=label,
                        grouping=grouping,
                        category_title=subset_title,
                    )
                    outputs.extend(_save_figure(
                        fig,
                        output_dir
                        / subset_name
                        / "process_closure"
                        / process
                        / grouping
                        / f"closure_{variable}",
                    ))

    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else output_dir / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(outputs, indent=2) + "\n")
    logger.info("Saved %d high-FF closure plot files to %s.", len(outputs), output_dir)
    return outputs


def create_fake_factor_plots(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    feature_paths: Iterable[Union[str, Path]],
    reduced_weight_paths: Iterable[Union[str, Path]],
    plotting_config_path: Union[str, Path],
    labels_path: Union[str, Path],
    output_dir: Union[str, Path],
    manifest_path: Union[str, Path, None] = None,
    variable_set: str = "variables_set_small",
    channel: str = "et",
    feature_suffix: str = "",
) -> list[str]:
    """Create workflow-rooted closure and FF-distribution plots."""
    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    plotting_config = _read_yaml(plotting_config_path)
    labels = _read_channel_labels(labels_path, channel)
    variables = plotting_config.get(variable_set, [])
    binning = plotting_config.get("bins_by_variable", {})
    if not variables:
        raise ValueError(f"No plotting variables configured in {variable_set}.")

    df = load_data(data_path, masks_path)
    for feature_path in (*feature_paths, *reduced_weight_paths):
        df.load_feature_file(feature_path)
    _install_feature_aliases(df, "ff_dnn", feature_suffix)
    required_features = (
        "ff_classic",
        "reduced_weight_wjets_tau_decaymode_2_nominal",
        "reduced_weight_wjets_tau_decaymode_2_alt_nominal",
        "reduced_weight_wjets_njets_nominal",
        "reduced_weight_qcd_tau_decaymode_2_nominal",
        "reduced_weight_qcd_tau_decaymode_2_alt_nominal",
        "reduced_weight_qcd_njets_nominal",
    )
    missing = [
        feature
        for feature in required_features
        if feature not in df.events.columns
    ]
    if missing:
        raise KeyError(
            f"Fake-factor feature file is missing columns: {missing}"
        )
    _require_grouped_fake_factor_features(df)

    outputs = []
    grouping_names = GROUPING_NAMES
    process_closures = {
        "wjets": FF_closure_in_DR_wjets,
        "qcd": FF_closure_in_DR_qcd,
        "ttbar": FF_closure_in_DR_ttbar,
    }

    for subset_name, subset_title, selection in (
        *PLOT_SUBSETS,
        *CLOSURE_ONLY_SUBSETS,
    ):
        subset_df = _plot_subset(df, selection)
        logger.info(
            "Creating fake-factor plots for subset %s with %d events.",
            subset_name,
            len(subset_df.events),
        )
        subset_output_dir = output_dir / subset_name

        for grouping in grouping_names:
            for variable in variables:
                if variable not in binning:
                    raise KeyError(f"No binning configured for {variable}.")
                bins = _plot_bins(binning[variable])
                label = labels.get(variable, variable)

                fig, _, _ = plot_closure(
                    df=subset_df,
                    var=variable,
                    bins=bins,
                    label=label,
                    grouping=grouping,
                    plot_classic_ff_comp=True,
                    plot_corr_hline=False,
                )
                outputs.extend(_save_figure(
                    fig,
                    subset_output_dir
                    / "closure"
                    / grouping
                    / f"closure_{variable}",
                ))

                for process, plot_function in process_closures.items():
                    fig, _ = plot_function(
                        df=subset_df,
                        var=variable,
                        bins=bins,
                        label=label,
                        grouping=grouping,
                    )
                    outputs.extend(_save_figure(
                        fig,
                        subset_output_dir
                        / "process_closure"
                        / process
                        / grouping
                        / f"closure_{variable}",
                    ))

            if selection is not None and selection[1] == "isin":
                continue

            category_title = (
                f"{subset_title}, split in {_grouping_title(grouping)}"
            )
            fig, _ = plot_fake_factors_grouped(
                df=subset_df,
                category_title=category_title,
                grouping=grouping,
            )
            outputs.extend(_save_figure(
                fig,
                subset_output_dir
                / "distribution"
                / "AR"
                / f"fake_factors_{grouping}",
            ))

            fig, _ = plot_fake_factors_in_dr_grouped(
                df=subset_df,
                category_title=category_title,
                grouping=grouping,
            )
            outputs.extend(_save_figure(
                fig,
                subset_output_dir
                / "distribution"
                / "AR_like"
                / f"fake_factors_{grouping}",
            ))

    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else output_dir / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(outputs, indent=2) + "\n")
    logger.info("Saved %d fake-factor plot files to %s.", len(outputs), output_dir)
    return outputs


def create_fake_factor_opposite_grouping_distribution_plots(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    feature_path: Union[str, Path],
    output_dir: Union[str, Path],
    manifest_path: Union[str, Path, None] = None,
    feature_suffix: str = "",
) -> list[str]:
    """Create inclusive GroupedDNN FF distributions split by the other axis."""
    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    df = load_data(data_path, masks_path)
    df.load_feature_file(feature_path)
    _install_feature_aliases(df, "ff_dnn", feature_suffix)
    _require_grouped_fake_factor_features(df)

    outputs = []
    plot_functions = {
        "AR": plot_fake_factors_grouped,
        "AR_like": plot_fake_factors_in_dr_grouped,
    }
    for grouping in GROUPING_NAMES:
        split_grouping = _opposite_distribution_grouping(grouping)
        logger.info(
            "Creating inclusive grouped-DNN FF distribution plots for FF "
            "grouping %s split in %s with %d events.",
            grouping,
            split_grouping,
            len(df.events),
        )
        for region_name, plot_function in plot_functions.items():
            fig, _ = plot_function(
                df=df,
                category_title=(
                    f"inclusive, {_grouping_title(grouping)} FF, split in "
                    f"{_grouping_title(split_grouping)}"
                ),
                grouping=grouping,
                split_grouping=split_grouping,
            )
            outputs.extend(_save_figure(
                fig,
                output_dir
                / "inclusive"
                / region_name
                / f"fake_factors_{grouping}_split_{split_grouping}",
            ))

    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else output_dir / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(outputs, indent=2) + "\n")
    logger.info(
        "Saved %d grouped-DNN FF distribution split plot files to %s.",
        len(outputs),
        output_dir,
    )
    return outputs


create_fake_factor_distribution_split_plots = (
    create_fake_factor_opposite_grouping_distribution_plots
)


def create_corrected_fake_factor_closure_plots(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    corrected_feature_path: Union[str, Path],
    classic_feature_path: Union[str, Path],
    plotting_config_path: Union[str, Path],
    labels_path: Union[str, Path],
    output_dir: Union[str, Path],
    manifest_path: Union[str, Path, None] = None,
    variable_set: str = "variables_set_small",
    channel: str = "et",
    feature_suffix: str = "",
) -> list[str]:
    """Create closure plots using the corrected combined DNN fake factors."""
    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    plotting_config = _read_yaml(plotting_config_path)
    labels = _read_channel_labels(labels_path, channel)
    variables = plotting_config.get(variable_set, [])
    binning = plotting_config.get("bins_by_variable", {})
    if not variables:
        raise ValueError(f"No plotting variables configured in {variable_set}.")

    df = load_data(data_path, masks_path)
    df.load_feature_file(corrected_feature_path)
    df.load_feature_file(classic_feature_path)

    for grouping in GROUPING_NAMES:
        grouping_part = grouping_suffix(grouping)
        corrected_name = (
            f"ff_dnn_corrected{grouping_part}{feature_suffix}"
        )
        if corrected_name not in df.events.columns:
            raise KeyError(
                f"Corrected fake-factor feature is missing {corrected_name}."
            )
        df.events[f"ff_dnn{grouping_part}"] = df.events[corrected_name]

    outputs = []
    for subset_name, _, selection in (
        *PLOT_SUBSETS,
        *CLOSURE_ONLY_SUBSETS,
    ):
        subset_df = _plot_subset(df, selection)
        subset_output_dir = output_dir / subset_name
        logger.info(
            "Creating corrected fake-factor closure plots for subset %s "
            "with %d events.",
            subset_name,
            len(subset_df.events),
        )
        for grouping in GROUPING_NAMES:
            for variable in variables:
                if variable not in binning:
                    raise KeyError(f"No binning configured for {variable}.")
                fig, _, _ = plot_closure(
                    df=subset_df,
                    var=variable,
                    bins=_plot_bins(binning[variable]),
                    label=labels.get(variable, variable),
                    grouping=grouping,
                    plot_classic_ff_comp=True,
                    plot_corr_hline=False,
                )
                outputs.extend(_save_figure(
                    fig,
                    subset_output_dir
                    / "closure"
                    / grouping
                    / f"closure_{variable}",
                ))

    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else output_dir / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(outputs, indent=2) + "\n")
    logger.info(
        "Saved %d corrected fake-factor closure plot files to %s.",
        len(outputs),
        output_dir,
    )
    return outputs


def create_mlf_closure_plots(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    mlf_feature_path: Union[str, Path],
    classic_feature_path: Union[str, Path],
    plotting_config_path: Union[str, Path],
    labels_path: Union[str, Path],
    output_dir: Union[str, Path],
    mlf_column: str,
    manifest_path: Union[str, Path, None] = None,
    variable_set: str = "variables_set_small",
    channel: str = "et",
) -> list[str]:
    """Create inclusive and njets-split closure plots for an MLF FF column."""
    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    plotting_config = _read_yaml(plotting_config_path)
    labels = _read_channel_labels(labels_path, channel)
    variables = plotting_config.get(variable_set, [])
    binning = plotting_config.get("bins_by_variable", {})
    if not variables:
        raise ValueError(f"No plotting variables configured in {variable_set}.")

    df = load_data(data_path, masks_path)
    df.load_feature_file(mlf_feature_path)
    df.load_feature_file(classic_feature_path)
    if mlf_column not in df.events.columns:
        raise KeyError(f"MLF fake-factor feature is missing {mlf_column}.")

    df.events["ff_dnn_njets"] = df.events[mlf_column]

    outputs = []
    closure_subsets = (
        ("inclusive", "inclusive", None),
        ("njets_eq_0", r"$N_{\mathrm{jets}} = 0$", ("njets", "eq", 0)),
        ("njets_eq_1", r"$N_{\mathrm{jets}} = 1$", ("njets", "eq", 1)),
        ("njets_ge_2", r"$N_{\mathrm{jets}} \geq 2$", ("njets", "ge", 2)),
    )
    for subset_name, _, selection in closure_subsets:
        subset_df = _plot_subset(df, selection)
        subset_output_dir = output_dir / subset_name
        logger.info(
            "Creating MLF closure plots for subset %s with %d events.",
            subset_name,
            len(subset_df.events),
        )
        for variable in variables:
            if variable not in binning:
                raise KeyError(f"No binning configured for {variable}.")
            fig, _, _ = plot_closure(
                df=subset_df,
                var=variable,
                bins=_plot_bins(binning[variable]),
                label=labels.get(variable, variable),
                grouping="njets",
                plot_classic_ff_comp=True,
                plot_corr_hline=False,
            )
            outputs.extend(_save_figure(
                fig,
                subset_output_dir
                / "closure"
                / "njets"
                / f"closure_{variable}",
            ))

    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else output_dir / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(outputs, indent=2) + "\n")
    logger.info("Saved %d MLF closure plot files to %s.", len(outputs), output_dir)
    return outputs


def create_high_fake_factor_distribution_plots(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    feature_path: Union[str, Path],
    output_dir: Union[str, Path],
    manifest_path: Union[str, Path, None] = None,
    value_min: float = 1.0,
    value_max: float = 100.0,
    n_bins: int = 90,
    feature_suffix: str = "",
) -> list[str]:
    """Plot grouped AR and AR-like FF distributions in a selected range."""
    if value_max <= value_min:
        raise ValueError("value_max must be greater than value_min.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    df = load_data(data_path, masks_path)
    df.load_feature_file(feature_path)
    _install_feature_aliases(df, "ff_dnn", feature_suffix)

    required_features = (
        "ff_dnn_wjets",
        "ff_dnn_qcd",
        "ff_dnn_ttbar",
        "ff_dnn_wjets_tau_decaymode_2_alt",
        "ff_dnn_qcd_tau_decaymode_2_alt",
        "ff_dnn_ttbar_tau_decaymode_2_alt",
        "ff_dnn_wjets_njets",
        "ff_dnn_qcd_njets",
        "ff_dnn_ttbar_njets",
    )
    missing = [
        feature
        for feature in required_features
        if feature not in df.events.columns
    ]
    if missing:
        raise KeyError(
            f"Fake-factor feature file is missing columns: {missing}"
        )

    outputs = []
    plot_functions = {
        "AR": plot_fake_factors_grouped_range,
        "AR_like": plot_fake_factors_in_dr_grouped_range,
    }
    for grouping in GROUPING_NAMES:
        for region_name, plot_function in plot_functions.items():
            fig, _ = plot_function(
                df=df,
                category_title=(
                    f"{region_name.replace('_', '-')}, split in "
                    f"{_grouping_title(grouping)}"
                ),
                grouping=grouping,
                value_min=value_min,
                value_max=value_max,
                n_bins=n_bins,
            )
            outputs.extend(_save_figure(
                fig,
                output_dir
                / region_name
                / f"fake_factors_{grouping}_{value_min:g}_{value_max:g}",
            ))

    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else output_dir / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(outputs, indent=2) + "\n")
    logger.info(
        "Saved %d high-range fake-factor plot files to %s.",
        len(outputs),
        output_dir,
    )
    return outputs


def _plot_single_dnn_distributions(df, region, value_min, value_max, n_bins):
    bins = np.linspace(value_min, value_max, n_bins + 1)
    frames = (
        df.data.AR if region == "AR" else df.data.AR_like_wjets,
        df.data.AR if region == "AR" else df.data.AR_like_qcd,
        df.data.AR if region == "AR" else df.data.AR_like_ttbar,
    )
    columns = (
        "ff_dnn_single_wjets",
        "ff_dnn_single_qcd",
        "ff_dnn_single_ttbar",
    )
    process_titles = ("W+jets", "QCD", r"$t\bar{t}$")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(5, 7),
        sharex=True,
    )
    for index, (axis, frame, column, title) in enumerate(
        zip(axes, frames, columns, process_titles)
    ):
        counts, _, _ = axis.hist(
            frame[column],
            bins=bins,
            histtype="step",
            linewidth=2,
        )
        axis.set_ylabel("Events")
        axis.text(
            0.98,
            0.88,
            title,
            ha="right",
            va="top",
            transform=axis.transAxes,
            fontsize=11,
        )
        axis.tick_params(labelbottom=index == len(axes) - 1)
        maximum = np.max(counts) if len(counts) else 0
        axis.set_ylim(0, 1.2 * maximum if maximum > 0 else 1)
    axes[-1].set_xlabel(r"$F_{\mathrm{F}}$ value")
    axes[-1].set_xlim(value_min, value_max)
    CMS_CHANNEL_TITLE([axes[0]])
    CMS_LUMI_TITLE([axes[0]])
    CMS_LABEL([axes[0]])
    CMS_CATEGORY_TITLE([axes[0]], title=f"single DNN, {region}")
    return fig


def create_single_dnn_distribution_plots(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    feature_path: Union[str, Path],
    output_dir: Union[str, Path],
    manifest_path: Union[str, Path],
    value_min: float,
    value_max: float,
    n_bins: int,
    feature_suffix: str = "",
) -> list[str]:
    if value_max <= value_min:
        raise ValueError("value_max must be greater than value_min.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    df = load_data(data_path, masks_path)
    df.load_feature_file(feature_path)
    _install_feature_aliases(df, "ff_dnn_single", feature_suffix)
    required = (
        "ff_dnn_single_wjets",
        "ff_dnn_single_qcd",
        "ff_dnn_single_ttbar",
    )
    missing = [name for name in required if name not in df.events.columns]
    if missing:
        raise KeyError(f"Single-DNN feature file is missing: {missing}")

    outputs = []
    for region in ("AR", "AR_like"):
        fig = _plot_single_dnn_distributions(
            df,
            region,
            value_min,
            value_max,
            n_bins,
        )
        outputs.extend(_save_figure(
            fig,
            output_dir
            / region
            / f"fake_factors_single_dnn_{value_min:g}_{value_max:g}",
        ))

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(outputs, indent=2) + "\n")
    return outputs


def create_single_dnn_fake_factor_plots(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    feature_paths: Iterable[Union[str, Path]],
    reduced_weight_paths: Iterable[Union[str, Path]],
    plotting_config_path: Union[str, Path],
    labels_path: Union[str, Path],
    output_dir: Union[str, Path],
    manifest_path: Union[str, Path],
    reduced_weight_grouping: str,
    variable_set: str = "variables_set_small",
    channel: str = "et",
    feature_suffix: str = "",
) -> list[str]:
    """Create subset closure and inclusive distribution plots for single-DNN FFs."""
    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    plotting_config = _read_yaml(plotting_config_path)
    labels = _read_channel_labels(labels_path, channel)
    variables = plotting_config.get(variable_set, [])
    binning = plotting_config.get("bins_by_variable", {})
    if not variables:
        raise ValueError(f"No plotting variables configured in {variable_set}.")

    df = load_data(data_path, masks_path)
    for feature_path in (*feature_paths, *reduced_weight_paths):
        df.load_feature_file(feature_path)
    _install_feature_aliases(df, "ff_dnn_single", feature_suffix)

    suffix = grouping_suffix(reduced_weight_grouping)
    aliases = {
        f"ff_dnn{suffix}": "ff_dnn_single",
        **{
            f"ff_dnn_{process}{suffix}": f"ff_dnn_single_{process}"
            for process in ("wjets", "qcd", "ttbar")
        },
    }
    required_features = (
        "ff_classic",
        *aliases.values(),
        f"reduced_weight_wjets_{reduced_weight_grouping}_nominal",
        f"reduced_weight_qcd_{reduced_weight_grouping}_nominal",
    )
    missing = [
        feature
        for feature in required_features
        if feature not in df.events.columns
    ]
    if missing:
        raise KeyError(
            f"Single-DNN plotting feature files are missing columns: {missing}"
        )
    for alias, source in aliases.items():
        df.events[alias] = df.events[source]

    outputs = []
    process_closures = {
        "wjets": FF_closure_in_DR_wjets,
        "qcd": FF_closure_in_DR_qcd,
        "ttbar": FF_closure_in_DR_ttbar,
    }
    closure_subsets = (
        ("inclusive", "inclusive", None),
        *CLOSURE_ONLY_SUBSETS,
    )
    for subset_name, _, selection in closure_subsets:
        subset_df = _plot_subset(df, selection)
        subset_output_dir = (
            output_dir
            if selection is None
            else output_dir / subset_name
        )
        logger.info(
            "Creating single-DNN closure plots for subset %s with %d events.",
            subset_name,
            len(subset_df.events),
        )

        for variable in variables:
            if variable not in binning:
                raise KeyError(f"No binning configured for {variable}.")
            bins = _plot_bins(binning[variable])
            label = labels.get(variable, variable)

            fig, _, _ = plot_closure(
                df=subset_df,
                var=variable,
                bins=bins,
                label=label,
                grouping=reduced_weight_grouping,
                plot_classic_ff_comp=True,
                plot_corr_hline=False,
            )
            outputs.extend(_save_figure(
                fig,
                subset_output_dir / "closure" / f"closure_{variable}",
            ))

            for process, plot_function in process_closures.items():
                fig, _ = plot_function(
                    df=subset_df,
                    var=variable,
                    bins=bins,
                    label=label,
                    grouping=reduced_weight_grouping,
                )
                outputs.extend(_save_figure(
                    fig,
                    subset_output_dir
                    / "process_closure"
                    / process
                    / f"closure_{variable}",
                ))

    for region in ("AR", "AR_like"):
        fig = _plot_single_dnn_distributions(
            df,
            region,
            value_min=0.0,
            value_max=1.0,
            n_bins=100,
        )
        outputs.extend(_save_figure(
            fig,
            output_dir / "distribution" / region / "fake_factors_single_dnn",
        ))

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(outputs, indent=2) + "\n")
    logger.info(
        "Saved %d inclusive single-DNN fake-factor plot files to %s.",
        len(outputs),
        output_dir,
    )
    return outputs
