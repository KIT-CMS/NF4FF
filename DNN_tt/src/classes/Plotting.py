import math
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
from collections.abc import Iterable

def CMS_CHANNEL_TITLE(ax, *args, **kwargs):
    if isinstance(ax, Iterable):
        ax = ax[0]
    ax.set_title(
        r"$\mathrm{\tau_h\tau_h}$",
        #fontsize=20,
        loc="left",
        #fontproperties="Tex Gyre Heros"
    )

def CMS_CATEGORY_TITLE(ax, title="tau_DM: inclusive", *args, **kwargs):
    if isinstance(ax, Iterable):
        ax = ax[0]
    ax.set_title(
        title,
        #fontsize=10,
        loc="center",
        #fontproperties="Tex Gyre Heros"
    )

def CMS_LUMI_TITLE(ax, *args, **kwargs):
    if isinstance(ax, Iterable):
        ax = ax[0]
    ax.set_title(
        r"59.8 $\mathrm{fb}^{-1}$ (2018, 13 TeV)",
        #fontsize=20,
        loc="right",
        #fontproperties="Tex Gyre Heros"
    )

def CMS_LABEL(ax, *args, **kwargs):
    if isinstance(ax, Iterable):
        ax = ax[0]
    ax.text(
        0.025, 0.95,
        "Private work (CMS data/simulation)",
        fontsize=20,
        verticalalignment='top',
        style ="italic",
        fontproperties="Tex Gyre Heros:italic",
        bbox=dict(facecolor="white", alpha=0, edgecolor="white", boxstyle="round,pad=0.5"),
        transform=ax.transAxes
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
    grouping = None,
    corr_emb_ff = 1.0,
    plot_classic_ff_comp = False,
    plot_corr_hline = False,
):
    hep.style.use(hep.style.CMS)

    if grouping == 'tau_decaymode':
        ff_dnn_tau1 = 'ff_dnn_tau1_tau_dm'
        ff_dnn_tau2 = 'ff_dnn_tau2_tau_dm'
    elif grouping == 'njets':
        ff_dnn_tau1 = 'ff_dnn_tau1_njets'
        ff_dnn_tau2 = 'ff_dnn_tau2_njets'
    else:
        ff_dnn_tau1 = 'ff_dnn_tau1'
        ff_dnn_tau2 = 'ff_dnn_tau2'

    histograms = {}

    list_processes = [
        'data',
        'diboson',
        'DYjets',
        'ST',
        'ttbar',
        'embedding',
        'wjets',
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


    #jet_fakes_classic, var_jet_fakes_classic = estimate_jet_fakes(
    #    df,
    #    bins,
    #    var,
    #    'ff_classic',
    #)

    jet_fakes_dnn, var_jet_fakes_dnn = estimate_jet_fakes(
        df,
        bins,
        var,
        ff_dnn_tau1,
        ff_dnn_tau2,
    )

    #histograms['jet_fakes_classic'] = {
    #    'counts': jet_fakes_classic,
    #    'variance': var_jet_fakes_classic,
    #}

    histograms['jet_fakes_dnn'] = {
        'counts': jet_fakes_dnn,
        'variance': var_jet_fakes_dnn,
    }

    background_dnn = (
        histograms['diboson']['counts']
        + histograms['DYjets']['counts']
        + histograms['ST']['counts']
        + histograms['ttbar']['counts']
        + histograms['embedding']['counts']
        + histograms['wjets']['counts']
        + histograms['jet_fakes_dnn']['counts']
    )

    #background_classic = (
    #    histograms['diboson']['counts']
    #    + histograms['DYjets']['counts']
    #    + histograms['ST']['counts']
    #    + histograms['ttbar_L']['counts']
    #    + histograms['embedding']['counts']
    #    + histograms['jet_fakes_classic']['counts']
    #)

    variance_background_dnn = (
        histograms['diboson']['variance']
        + histograms['DYjets']['variance']
        + histograms['ST']['variance']
        + histograms['ttbar']['variance']
        + histograms['embedding']['variance']
        + histograms['wjets']['variance']
        + histograms['jet_fakes_dnn']['variance']
    )

    #variance_background_classic = (
    #    histograms['diboson']['variance']
    #    + histograms['DYjets']['variance']
    #    + histograms['ST']['variance']
    #    + histograms['ttbar_L']['variance']
    #    + histograms['embedding']['variance']
    #    + histograms['jet_fakes_classic']['variance']
    #)

    histograms['background_dnn'] = {
        'counts': background_dnn,
        'variance': variance_background_dnn,
    }

    #histograms['background_classic'] = {
    #    'counts': background_classic,
    #    'variance': variance_background_classic,
    #}


    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    err_bin = 0.5 * bin_widths


    err_data = np.sqrt(histograms['data']['variance'])


    err_stat_dnn = np.sqrt(
        histograms['background_dnn']['variance']
    )

    #err_stat_classic = np.sqrt(
    #    histograms['background_classic']['variance']
    #)


    err_stat_rel_dnn = np.divide(
        err_stat_dnn,
        histograms['background_dnn']['counts'],
        out=np.zeros_like(err_stat_dnn),
        where=histograms['background_dnn']['counts'] > 0,
    )

    #err_stat_rel_classic = np.divide(
    #    err_stat_classic,
    #    histograms['background_classic']['counts'],
    #    out=np.zeros_like(err_stat_classic),
    #    where=histograms['background_classic']['counts'] > 0,
    #)
    #if plot_classic_ff_comp == True:
    #    fig, ax = plt.subplots(
    #        4,
    #        1,
    #        figsize=(9, 9),
    #        sharex=True,
    #        gridspec_kw={
    #            'height_ratios': [4, 1, 0.2, 1],
    #            'hspace': 0.05,
    #        },
    #        constrained_layout=True,
    #    )
    #else:
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(11.7, 9.1),
        sharex=True,
        gridspec_kw={
            'height_ratios': [3, 1],
            'hspace': 0.05,
        },
        constrained_layout=True,
    )    

    stack_components = [
        (histograms['diboson']['counts'], "#94a4a2", 'Diboson'),
        (histograms['ttbar']['counts'], '#832db6', r'$t\bar{t} \to \tau$'),
        (histograms['ST']['counts'], "#717581", r"Single t"),
        (histograms['DYjets']['counts'], '#3f90da', r'$Z \to \ell \ell$'),
        (histograms['jet_fakes_dnn']['counts'], "#a96b59", r'Jet $\rightarrow \tau_h$'),
        (histograms['embedding']['counts'], '#ffa90e', r'$\tau$ embedded'),
        (histograms['wjets']['counts'], '#e76300', r"W+jets"),
    ]
    #print(histograms['wjets']['counts'])
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
    if plot_corr_hline:
        ax[1].axhline(1/corr_emb_ff, color='blue', linestyle='--', linewidth=1.5)
    ax[1].axhline(1, color='red', linestyle='--', linewidth=1.5)
    
    ax[1].set_ylabel("Data / Model", loc='center')
    ax[1].set_ylim([0.75, 1.25])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=2, frameon=False)

    if plot_classic_ff_comp:

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


def plot_closure_incl(
    df,
    var: str,
    bins: np.ndarray,
    label: str,
    grouping = None,
    corr_emb_ff = 1.0,
):
    hep.style.use(hep.style.CMS)

    if grouping == 'tau_decaymode':
        ff_dnn_tau1 = 'ff_dnn_tau1_tau_dm'
        ff_dnn_tau2 = 'ff_dnn_tau2_tau_dm'
    elif grouping == 'njets':
        ff_dnn_tau1 = 'ff_dnn_tau1_njets'
        ff_dnn_tau2 = 'ff_dnn_tau2_njets'
    else:
        ff_dnn = 'ff_dnn_incl'

    histograms = {}

    list_processes = [
        'data',
        'diboson',
        'DYjets',
        'ST',
        'ttbar',
        'embedding',
        'wjets',
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

    #todo
    jet_fakes_dnn, var_jet_fakes_dnn = estimate_jet_fakes_incl(
        df,
        bins,
        var,
        ff_dnn
    )


    histograms['jet_fakes_dnn'] = {
        'counts': jet_fakes_dnn,
        'variance': var_jet_fakes_dnn,
    }

    background_dnn = (
        histograms['diboson']['counts']
        + histograms['DYjets']['counts']
        + histograms['ST']['counts']
        + histograms['ttbar']['counts']
        + histograms['embedding']['counts']
        + histograms['wjets']['counts']
        + histograms['jet_fakes_dnn']['counts']
    )



    variance_background_dnn = (
        histograms['diboson']['variance']
        + histograms['DYjets']['variance']
        + histograms['ST']['variance']
        + histograms['ttbar']['variance']
        + histograms['embedding']['variance']
        + histograms['wjets']['variance']
        + histograms['jet_fakes_dnn']['variance']
    )


    histograms['background_dnn'] = {
        'counts': background_dnn,
        'variance': variance_background_dnn,
    }


    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    err_bin = 0.5 * bin_widths


    err_data = np.sqrt(histograms['data']['variance'])


    err_stat_dnn = np.sqrt(
        histograms['background_dnn']['variance']
    )

    err_stat_rel_dnn = np.divide(
        err_stat_dnn,
        histograms['background_dnn']['counts'],
        out=np.zeros_like(err_stat_dnn),
        where=histograms['background_dnn']['counts'] > 0,
    )


    fig, ax = plt.subplots(
        2,
        1,
        figsize=(11.7, 9.1),
        sharex=True,
        gridspec_kw={
            'height_ratios': [3, 1],
            'hspace': 0.05,
        },
        constrained_layout=True,
    )    

    stack_components = [
        (histograms['diboson']['counts'], "#94a4a2", 'Diboson'),
        (histograms['ttbar']['counts'], '#832db6', r'$t\bar{t} \to \tau$'),
        (histograms['ST']['counts'], "#717581", r"Single t"),
        (histograms['DYjets']['counts'], '#3f90da', r'$Z \to \ell \ell$'),
        (histograms['jet_fakes_dnn']['counts'], "#a96b59", r'Jet $\rightarrow \tau_h$'),
        (histograms['embedding']['counts'], '#ffa90e', r'$\tau$ embedded'),
        (histograms['wjets']['counts'], '#e76300', r"W+jets"),
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
    
    ax[1].axhline(1, color='red', linestyle='--', linewidth=1.5)
    
    ax[1].set_ylabel("Data / Model", loc='center')
    ax[1].set_ylim([0.75, 1.25])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=2, frameon=False)

    ax[1].set_xlabel(label)

    return fig, ax, histograms


def plot_closure_c(
    df,
    var: str,
    bins: np.ndarray,
    label: str,
    grouping = 'tau_decaymode',
    corr_emb_ff = 1.0,
    plot_classic_ff_comp = True,
    plot_corr_hline = True,
    squeeze_upper_bound = '0',
):
    if grouping == 'tau_decaymode':
        ff_dnn = f'ff_dnn_{squeeze_upper_bound}'

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


    jet_fakes_classic, var_jet_fakes_classic = estimate_jet_fakes(
        df,
        bins,
        var,
        'ff_classic',
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


def plot_fake_factors(
        df,
        category_title = None,
        clipped = True
) -> None:
    hep.style.use(hep.style.CMS)
	
    if clipped:
        ff_dnn_tau1 = 'ff_dnn_tau1'
        ff_dnn_tau2 = 'ff_dnn_tau2'
    
        bins_tau1 = np.linspace(0, 1, 50)
        bins_tau2 = np.linspace(0, 1., 50)
    else:
        ff_dnn_tau1 = 'ff_unclipped_dnn_tau1'
        ff_dnn_tau2 = 'ff_unclipped_dnn_tau2'
    
        bins_tau1 = np.linspace(0, 10, 50)
        bins_tau2 = np.linspace(0, 10, 50)
    

    fig, ax = plt.subplots(2, 1, figsize=(11.7, 9.1))

    ax[0].hist(df.data.AR_tau1[ff_dnn_tau1], bins=bins_tau1, histtype = 'step', linewidth = 2, label='Tau 1')
    #ax[0].hist(df.data.AR[ff_dnn_tau1][df.data.AR.tau_decaymode_2 == 1], bins=bins_tau1, histtype = 'step', ls = '--', label='Wjets: t_dm = 0')
    #ax[0].hist(df.data.AR[ff_dnn_tau1][df.data.AR.tau_decaymode_2 == 0], bins=bins_tau1, histtype = 'step', ls = '--', label='Wjets: t_dm = 1')
    #ax[0].hist(df.data.AR[ff_dnn_tau1][df.data.AR.tau_decaymode_2 == 10], bins=bins_tau1, histtype = 'step', ls = '--', label='Wjets: t_dm = 10')
    #ax[0].hist(df.data.AR[ff_dnn_tau1][df.data.AR.tau_decaymode_2 == 11], bins=bins_tau1, histtype = 'step', ls = '--', label='Wjets: t_dm = 11')
    ax[0].set_ylabel("Events")
    ax[0].legend()
    #ax[0].set_ylim(0, 33000)

    CMS_CHANNEL_TITLE([ax[0]])
    CMS_LUMI_TITLE([ax[0]])
    CMS_LABEL([ax[0]])
    CMS_CATEGORY_TITLE([ax[0]], title = category_title)

    ax[1].set_ylabel('Events')
    ax[1].hist(df.data.AR_tau2[ff_dnn_tau2], bins=bins_tau2, histtype = 'step', linewidth = 2, label="Tau 2")
    #ax[1].hist(df.data.AR[ff_dnn_tau2][df.data.AR.tau_decaymode_2 == 0], bins=bins_tau2, histtype = 'step', ls = '--', label = "QCD: t_dm = 0")
    #ax[1].hist(df.data.AR[ff_dnn_tau2][df.data.AR.tau_decaymode_2 == 1], bins=bins_tau2, histtype = 'step', ls = '--', label = "QCD: t_dm = 1")
    #ax[1].hist(df.data.AR[ff_dnn_tau2][df.data.AR.tau_decaymode_2 == 10], bins=bins_tau2, histtype = 'step', ls = '--', label = "QCD : t_dm = 10")
    #ax[1].hist(df.data.AR[ff_dnn_tau2][df.data.AR.tau_decaymode_2 == 11], bins=bins_tau2, histtype = 'step', ls = '--', label = "QCD : t_dm = 11")

    ax[1].set_xlabel("fake_factor")
    ax[1].legend()
    return fig, ax


def plot_fake_factors_incl(
        df,
        category_title = None,
        clipped = True
) -> None:
    hep.style.use(hep.style.CMS)
	
    if clipped:
        ff_dnn = 'ff_dnn_incl'
    
        bins = np.linspace(0, 1, 50)
    else:
        ff_dnn = 'ff_unclipped_dnn_incl'
    
        bins = np.linspace(0, 10, 50)
    

    fig, ax = plt.subplots(1, 1, figsize=(11.7, 9.1))
    
    CMS_CHANNEL_TITLE([ax])
    CMS_LUMI_TITLE([ax])
    CMS_LABEL([ax])
    CMS_CATEGORY_TITLE([ax], title = category_title)

    ax.hist(df.data.AR[ff_dnn], bins=bins, histtype = 'step', linewidth = 2, label='Tau incl.')
    ax.set_ylabel('Events')
    ax.set_xlabel("fake_factor")
    ax.legend()

    return fig, ax


def plot_classic_fake_factors(
        df,
        category_title = None,
        clipped = True
) -> None:
    hep.style.use(hep.style.CMS)
	
    if clipped:
        ff_tau1 = 'ff_classic_tau1'
        ff_tau2 = 'ff_classic_tau2'
    
        bins_tau1 = np.linspace(0.1, 0.2, 70)
        bins_tau2 = np.linspace(0.1, 0.2, 70)
    else:
        ff_tau1 = 'ff_unclipped_classic_tau1'
        ff_tau2 = 'ff_unclipped_classic_tau2'
    
        bins_tau1 = np.linspace(0, 10, 50)
        bins_tau2 = np.linspace(0, 10, 50)

    fig, ax = plt.subplots(2, 1, figsize=(11.7, 9.1))

    ax[0].hist(df.data.AR_tau1_jvoss[ff_tau1], bins=bins_tau1, histtype = 'step', linewidth = 2, label='Tau 1')
    #ax[0].hist(df.data.AR[ff_dnn_tau1][df.data.AR.tau_decaymode_2 == 1], bins=bins_tau1, histtype = 'step', ls = '--', label='Wjets: t_dm = 0')
    #ax[0].hist(df.data.AR[ff_dnn_tau1][df.data.AR.tau_decaymode_2 == 0], bins=bins_tau1, histtype = 'step', ls = '--', label='Wjets: t_dm = 1')
    #ax[0].hist(df.data.AR[ff_dnn_tau1][df.data.AR.tau_decaymode_2 == 10], bins=bins_tau1, histtype = 'step', ls = '--', label='Wjets: t_dm = 10')
    #ax[0].hist(df.data.AR[ff_dnn_tau1][df.data.AR.tau_decaymode_2 == 11], bins=bins_tau1, histtype = 'step', ls = '--', label='Wjets: t_dm = 11')
    ax[0].set_ylabel("Events")
    ax[0].legend()
    #ax[0].set_ylim(0, 33000)

    CMS_CHANNEL_TITLE([ax[0]])
    CMS_LUMI_TITLE([ax[0]])
    CMS_LABEL([ax[0]])
    CMS_CATEGORY_TITLE([ax[0]], title = category_title)

    ax[1].set_ylabel('Events')
    ax[1].hist(df.data.AR_tau2_jvoss[ff_tau2], bins=bins_tau2, histtype = 'step', linewidth = 2, label="Tau 2")
    #ax[1].hist(df.data.AR[ff_dnn_tau2][df.data.AR.tau_decaymode_2 == 0], bins=bins_tau2, histtype = 'step', ls = '--', label = "QCD: t_dm = 0")
    #ax[1].hist(df.data.AR[ff_dnn_tau2][df.data.AR.tau_decaymode_2 == 1], bins=bins_tau2, histtype = 'step', ls = '--', label = "QCD: t_dm = 1")
    #ax[1].hist(df.data.AR[ff_dnn_tau2][df.data.AR.tau_decaymode_2 == 10], bins=bins_tau2, histtype = 'step', ls = '--', label = "QCD : t_dm = 10")
    #ax[1].hist(df.data.AR[ff_dnn_tau2][df.data.AR.tau_decaymode_2 == 11], bins=bins_tau2, histtype = 'step', ls = '--', label = "QCD : t_dm = 11")

    ax[1].set_xlabel("fake_factor")
    ax[1].legend()
    return fig, ax


def plot_fake_factors_in_DR(
        df,
        category_title,
) -> None:

	bins_wjets = np.linspace(0, 1, 50)
	bins_qcd = np.linspace(0, 0.5, 50)

	fig, ax = plt.subplots(2, 1, figsize=(10, 7))
	ax[0].hist(df.data.AR_like_wjets.ff_dnn_wjets, bins=bins_wjets, histtype = 'step', linewidth = 2, label='Wjets: t_dm incl')
	ax[0].hist(df.data.AR_like_wjets.ff_dnn_wjets[df.data.AR_like_wjets.tau_decaymode_2 == 1], bins=bins_wjets, histtype = 'step', ls = '--', label='Wjets: t_dm = 0')
	ax[0].hist(df.data.AR_like_wjets.ff_dnn_wjets[df.data.AR_like_wjets.tau_decaymode_2 == 0], bins=bins_wjets, histtype = 'step', ls = '--', label='Wjets: t_dm = 1')
	ax[0].hist(df.data.AR_like_wjets.ff_dnn_wjets[df.data.AR_like_wjets.tau_decaymode_2 == 10], bins=bins_wjets, histtype = 'step', ls = '--', label='Wjets: t_dm = 10')
	ax[0].hist(df.data.AR_like_wjets.ff_dnn_wjets[df.data.AR_like_wjets.tau_decaymode_2 == 11], bins=bins_wjets, histtype = 'step', ls = '--', label='Wjets: t_dm = 11')
	ax[0].set_ylabel("Events")
	ax[0].legend()
	ax[0].set_ylim(0, 20000)

	CMS_CHANNEL_TITLE([ax[0]])
	CMS_LUMI_TITLE([ax[0]])
	CMS_LABEL([ax[0]])
	CMS_CATEGORY_TITLE([ax[0]], title = category_title)

	ax[1].set_ylabel('Events')
	ax[1].hist(df.data.AR_like_qcd.ff_dnn_qcd, bins=bins_qcd, histtype = 'step', linewidth = 2, label="QCD: t_dm: incl")
	ax[1].hist(df.data.AR_like_qcd.ff_dnn_qcd[df.data.AR_like_qcd.tau_decaymode_2 == 0], bins=bins_qcd, histtype = 'step', ls = '--', label = "QCD: t_dm = 0")
	ax[1].hist(df.data.AR_like_qcd.ff_dnn_qcd[df.data.AR_like_qcd.tau_decaymode_2 == 1], bins=bins_qcd, histtype = 'step', ls = '--', label = "QCD: t_dm = 1")
	ax[1].hist(df.data.AR_like_qcd.ff_dnn_qcd[df.data.AR_like_qcd.tau_decaymode_2 == 10], bins=bins_qcd, histtype = 'step', ls = '--', label = "QCD : t_dm = 10")
	ax[1].hist(df.data.AR_like_qcd.ff_dnn_qcd[df.data.AR_like_qcd.tau_decaymode_2 == 11], bins=bins_qcd, histtype = 'step', ls = '--', label = "QCD : t_dm = 11")

	ax[1].set_xlabel("fake_factor")
	ax[1].legend()
	return fig, ax


def FF_closure(
    data,
    data_weights,
	closure,
    closure_weights,
	bins,
	label,
	grouping = 'tau_decaymode',
    closure_labels=None,
    colors=None,
    main_linestyles=None,
    ratio_linestyles=None,
):
    def _as_list(value):
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _broadcast_style(value, length):
        if value is None:
            return ['--'] * length
        values = _as_list(value)
        if len(values) == 1:
            return values * length
        if len(values) != length:
            raise ValueError("style arguments must either be a single value or match the number of closure series")
        return values

    def _broadcast_colors(value, length):
        if value is None:
            return None
        values = _as_list(value)
        if len(values) == 1:
            return values * length
        if len(values) != length:
            raise ValueError("colors must either be a single value or match the number of closure series")
        return values

    closure_list = _as_list(closure)
    closure_weights_list = _as_list(closure_weights)

    if len(closure_list) != len(closure_weights_list):
        raise ValueError(
            "closure and closure_weights must have the same length when passing sequences"
        )

    if closure_labels is None:
        if len(closure_list) == 1:
            closure_labels = [r'$F_\mathrm{F} \cdot $ data(AR-like)']
        else:
            closure_labels = [
                rf'$F_\mathrm{{F}} \cdot $ data(AR-like) {idx + 1}'
                for idx in range(len(closure_list))
            ]
    elif isinstance(closure_labels, str):
        closure_labels = [closure_labels]
    elif len(closure_labels) != len(closure_list):
        raise ValueError("closure_labels must match the number of closure series")

    main_linestyles = _broadcast_style(main_linestyles, len(closure_list))
    ratio_linestyles = _broadcast_style(ratio_linestyles, len(closure_list))
    colors = _broadcast_colors(colors, len(closure_list))

    counts_data, bin_edges = np.histogram(data, weights=data_weights, bins=bins)
    variance_data, _ = np.histogram(data, weights=data_weights**2, bins=bins)

    err_data = np.sqrt(variance_data)
    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    err_bin = 0.5 * bin_widths

    closure_results = []
    for cls, wghts, series_label in zip(closure_list, closure_weights_list, closure_labels):
        cls = np.asarray(cls)
        wghts = np.asarray(wghts, dtype=float)

        counts_closure, _ = np.histogram(cls, weights=wghts, bins=bins)
        variance_closure, _ = np.histogram(cls, weights=wghts**2, bins=bins)
        err_closure = np.sqrt(variance_closure)

        closure_results.append(
            {
                "counts": counts_closure,
                "err": err_closure,
                "label": series_label,
            }
        )

    counts_closure_ref = closure_results[0]["counts"]

    data_ratio = np.full_like(counts_data, np.nan, dtype=float)
    data_ratio_err = np.full_like(err_data, np.nan, dtype=float)
    valid_data_ratio_bins = (counts_data > 0) & (counts_closure_ref > 0)
    data_ratio[valid_data_ratio_bins] = counts_data[valid_data_ratio_bins] / counts_closure_ref[valid_data_ratio_bins]
    data_ratio_err[valid_data_ratio_bins] = err_data[valid_data_ratio_bins] / counts_closure_ref[valid_data_ratio_bins]

    for result in closure_results:
        valid_closure_ratio_bins = counts_closure_ref > 0
        ratio_closure = np.full_like(counts_data, np.nan, dtype=float)
        ratio_err_closure = np.full_like(err_data, np.nan, dtype=float)
        ratio_closure[valid_closure_ratio_bins] = result["counts"][valid_closure_ratio_bins] / counts_closure_ref[valid_closure_ratio_bins]
        ratio_err_closure[valid_closure_ratio_bins] = result["err"][valid_closure_ratio_bins] / counts_closure_ref[valid_closure_ratio_bins]
        result["ratio_closure"] = ratio_closure
        result["ratio_err_closure"] = ratio_err_closure

    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    default_colors = prop_cycle.by_key().get("color", ["C0"]) if prop_cycle is not None else ["C0"]
    draw_order = list(range(len(closure_results)))
    if len(draw_order) > 1:
        draw_order = draw_order[1:] + draw_order[:1]

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={
            'height_ratios': [4, 1],
            'hspace': 0.05,
        },
        constrained_layout=True,
    )
    CMS_LABEL(ax)
    #CMS_CATEGORY_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_CHANNEL_TITLE(ax)

    data_counts_masked = counts_data.astype(float).copy()
    data_err_masked = err_data.astype(float).copy()
    data_counts_masked[counts_data <= 0] = np.nan
    data_err_masked[counts_data <= 0] = np.nan

    ax[0].errorbar(
        bin_centers,
        data_counts_masked,
        yerr=data_err_masked,
        xerr=err_bin,
        fmt='o',
        color='black',
        label='data',
        markersize=6,
        elinewidth=1.2,
        capsize=0,
    )

    for idx in draw_order:
        result = closure_results[idx]
        color = colors[idx] if colors is not None else default_colors[idx % len(default_colors)]
        ax[0].stairs(
            result["counts"],
            bin_edges,
            label=result["label"],
            ls=main_linestyles[idx],
            linewidth=2,
            color=color,
            alpha = 0.7,
        )


    ax[0].set_ylabel('Events')
    ax[0].legend()

    panel_max = np.nanmax([np.nanmax(data_counts_masked)] + [np.max(result["counts"]) for result in closure_results])
    ax[0].set_ylim(0, 1.15 * panel_max if np.isfinite(panel_max) and panel_max > 0 else 1.0)

    ax[1].errorbar(
        bin_centers,
        data_ratio,
        xerr=err_bin,
        yerr=data_ratio_err,
        fmt='o',
        color='black',
        markersize=6,
        elinewidth=1.2,
        capsize=0,
        label='data',
    )

    for idx in draw_order:
        result = closure_results[idx]
        color = colors[idx] if colors is not None else default_colors[idx % len(default_colors)]
        ratio_linestyle = ratio_linestyles[idx]
        ax[1].stairs(
            result["ratio_closure"],
            bin_edges,
            color=color,
            label=result["label"],
            ls=ratio_linestyle,
        )
        # ax[1].fill_between(
        #     bin_centers,
        #     1 - result["ratio_err_closure"],
        #     1 + result["ratio_err_closure"],
        #     color=color,
        #     alpha=0.15,
        #     step='mid',
        # )
    ax[1].set_ylabel("Data / Model")
    ax[1].set_ylim([0.75, 1.25])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    # ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=3, frameon=False)
    ax[1].set_xlabel(label)

    return fig, ax


def FF_closure_in_DR_wjets(
    df,
	var,
	bins,
	label,
	grouping = 'tau_decaymode',
):
    if grouping == 'tau_decaymode':
        ff_dnn_wjets = 'ff_dnn_wjets'
    elif grouping == 'njets':
        ff_dnn_wjets = 'ff_dnn_wjets_njets'

    counts_SR_like, bin_edges = np.histogram(df.data.SR_like_wjets[var], weights = df.data.SR_like_wjets.weight_wjets, bins = bins)
    counts_FF_AR_like, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * df.data.AR_like_wjets[ff_dnn_wjets], bins = bins)

    variance_SR_like, _ = np.histogram(df.data.SR_like_wjets[var], weights = df.data.SR_like_wjets.weight_wjets**2, bins = bins)
    variance_FF_AR_like, _ = np.histogram(
        df.data.AR_like_wjets[var], 
        weights = (df.data.AR_like_wjets.weight_wjets * df.data.AR_like_wjets[ff_dnn_wjets])**2,
        bins = bins)

    err_SR_like = np.sqrt(variance_SR_like)
    err_FF_AR_like = np.sqrt(variance_FF_AR_like)

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    err_bin = 0.5 * bin_widths


    fig, ax = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={
            'height_ratios': [4, 1],
            'hspace': 0.05,
        },
        constrained_layout=True,
    )
    CMS_LABEL(ax)
    #CMS_CATEGORY_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_CHANNEL_TITLE(ax)

    ax[0].errorbar(
        bin_centers,
        counts_SR_like,
        yerr=err_SR_like,
        xerr=err_bin,
        fmt='o',
        color='black',
        label='data(SR-like)',
        markersize=6,
        elinewidth=1.2,
        capsize=0,
    )

    ax[0].stairs(counts_FF_AR_like, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like)', ls = '--', linewidth = 2)


    ax[0].set_ylabel('Events')
    ax[0].legend()
    adjust_ylim_for_legend(ax[0])
    ratio = np.divide(counts_SR_like, counts_FF_AR_like, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like > 0)
    ratio_err_SR_like = np.divide(err_SR_like, counts_FF_AR_like, out=np.zeros_like(err_SR_like), where=counts_FF_AR_like > 0)
    ratio_err_FF_AR_like = np.divide(err_FF_AR_like, counts_FF_AR_like, out=np.zeros_like(err_FF_AR_like), where=counts_FF_AR_like > 0)

    ax[1].errorbar(bin_centers, ratio, xerr = err_bin, yerr = ratio_err_SR_like, fmt='o', color='black', markersize=6, label='ratio')
    ax[1].fill_between(
        bin_centers,
        1 - ratio_err_FF_AR_like,
        1 + ratio_err_FF_AR_like,
        color='gray',
        alpha=0.3,
        step='mid',
        label='Sys. Unc.',
    )
    ax[1].set_ylabel("Data / Model")
    ax[1].set_ylim([0.75, 1.25])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=2, frameon=False)
    ax[1].set_xlabel(label)

    return fig, ax

def FF_closure_in_DR_wjets_with_stat_unc(
    df,
	var,
	bins,
	label,
	grouping = 'tau_decaymode',
):
    ff_nominal = df.data.AR_like_wjets.ff_dnn_wjets
    ff_unc = df.data.AR_like_wjets.ff_wjets_unc
    ff_up = ff_nominal + ff_unc
    ff_down = ff_nominal - ff_unc

    counts_SR_like, bin_edges = np.histogram(df.data.SR_like_wjets[var], weights = df.data.SR_like_wjets.weight_wjets, bins = bins)
    counts_FF_AR_like_nominal, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_nominal, bins = bins)
    counts_FF_AR_like_up, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_up, bins = bins)
    counts_FF_AR_like_down, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_down, bins = bins)
    variance_SR_like, _ = np.histogram(df.data.SR_like_wjets[var], weights = df.data.SR_like_wjets.weight_wjets**2, bins = bins)
    variance_FF_AR_like, _ = np.histogram(
        df.data.AR_like_wjets[var], 
        weights = (df.data.AR_like_wjets.weight_wjets * ff_nominal)**2,
        bins = bins)

    err_SR_like = np.sqrt(variance_SR_like)
    err_FF_AR_like = np.sqrt(variance_FF_AR_like)

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    err_bin = 0.5 * bin_widths


    fig, ax = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={
            'height_ratios': [4, 1],
            'hspace': 0.05,
        },
        constrained_layout=True,
    )
    CMS_LABEL(ax)
    #CMS_CATEGORY_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_CHANNEL_TITLE(ax)

    ax[0].errorbar(
        bin_centers,
        counts_SR_like,
        yerr=err_SR_like,
        xerr=err_bin,
        fmt='o',
        color='black',
        label='data(SR-like)',
        markersize=6,
        elinewidth=1.2,
        capsize=0,
    )

    ax[0].stairs(counts_FF_AR_like_nominal, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like) nominal', ls = '-', linewidth = 2)
    ax[0].stairs(counts_FF_AR_like_up, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like) up', ls = '--', linewidth = 2)
    ax[0].stairs(counts_FF_AR_like_down, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like) dowm', ls = '--', linewidth = 2)

    ax[0].set_ylabel('Events')
    ax[0].legend()
    adjust_ylim_for_legend(ax[0])
    ratio = np.divide(counts_SR_like, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
    ratio_err_SR_like = np.divide(err_SR_like, counts_FF_AR_like_nominal, out=np.zeros_like(err_SR_like), where=counts_FF_AR_like_nominal > 0)
    ratio_err_FF_AR_like = np.divide(err_FF_AR_like, counts_FF_AR_like_nominal, out=np.zeros_like(err_FF_AR_like), where=counts_FF_AR_like_nominal > 0)

    ax[1].errorbar(bin_centers, ratio, xerr = err_bin, yerr = ratio_err_SR_like, fmt='o', color='black', markersize=6, label='ratio')
    ax[1].fill_between(
        bin_centers,
        1 - ratio_err_FF_AR_like,
        1 + ratio_err_FF_AR_like,
        color='gray',
        alpha=0.3,
        step='mid',
        label='Sys. Unc.',
    )
    ax[1].set_ylabel("Data / Model")
    ax[1].set_ylim([0.75, 1.25])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=2, frameon=False)
    ax[1].set_xlabel(label)

    return fig, ax


def FF_closure_in_DR_wjets_with_stat_unc(
    df,
	var,
	bins,
	label,
	grouping = 'tau_decaymode',
):
    ff_nominal = df.data.AR_like_wjets.ff_dnn_wjets
    ff_unc = df.data.AR_like_wjets.ff_wjets_unc
    ff_up = ff_nominal + ff_unc
    ff_down = ff_nominal - ff_unc

    counts_SR_like, bin_edges = np.histogram(df.data.SR_like_wjets[var], weights = df.data.SR_like_wjets.weight_wjets, bins = bins)
    counts_FF_AR_like_nominal, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_nominal, bins = bins)
    counts_FF_AR_like_up, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_up, bins = bins)
    counts_FF_AR_like_down, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_down, bins = bins)
    variance_SR_like, _ = np.histogram(df.data.SR_like_wjets[var], weights = df.data.SR_like_wjets.weight_wjets**2, bins = bins)
    variance_FF_AR_like, _ = np.histogram(
        df.data.AR_like_wjets[var], 
        weights = (df.data.AR_like_wjets.weight_wjets * ff_nominal)**2,
        bins = bins)

    err_SR_like = np.sqrt(variance_SR_like)
    err_FF_AR_like = np.sqrt(variance_FF_AR_like)

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    err_bin = 0.5 * bin_widths


    fig, ax = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={
            'height_ratios': [4, 1],
            'hspace': 0.05,
        },
        constrained_layout=True,
    )
    CMS_LABEL(ax)
    #CMS_CATEGORY_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_CHANNEL_TITLE(ax)

    ax[0].errorbar(
        bin_centers,
        counts_SR_like,
        yerr=err_SR_like,
        xerr=err_bin,
        fmt='o',
        color='black',
        label='data(SR-like)',
        markersize=6,
        elinewidth=1.2,
        capsize=0,
    )

    ax[0].stairs(counts_FF_AR_like_nominal, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like) nominal', ls = '-', linewidth = 2)
    ax[0].stairs(counts_FF_AR_like_up, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like) up', ls = '--', linewidth = 2)
    ax[0].stairs(counts_FF_AR_like_down, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like) dowm', ls = '--', linewidth = 2)

    ax[0].set_ylabel('Events')
    ax[0].legend()
    adjust_ylim_for_legend(ax[0])
    ratio = np.divide(counts_SR_like, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
    ratio_err_SR_like = np.divide(err_SR_like, counts_FF_AR_like_nominal, out=np.zeros_like(err_SR_like), where=counts_FF_AR_like_nominal > 0)
    ratio_err_FF_AR_like = np.divide(err_FF_AR_like, counts_FF_AR_like_nominal, out=np.zeros_like(err_FF_AR_like), where=counts_FF_AR_like_nominal > 0)

    ax[1].errorbar(bin_centers, ratio, xerr = err_bin, yerr = ratio_err_SR_like, fmt='o', color='black', markersize=6, label='ratio')
    ax[1].fill_between(
        bin_centers,
        1 - ratio_err_FF_AR_like,
        1 + ratio_err_FF_AR_like,
        color='gray',
        alpha=0.3,
        step='mid',
        label='Sys. Unc.',
    )
    ax[1].set_ylabel("Data / Model")
    ax[1].set_ylim([0.75, 1.25])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=2, frameon=False)
    ax[1].set_xlabel(label)

    return fig, ax

def FF_closure_in_DR_tau1(
    df,
	var,
	bins,
	label,
	grouping = 'njets',
):
    hep.style.use(hep.style.CMS)

    if grouping == 'tau_decaymode':
        ff_dnn_tau1 = 'ff_dnn_tau1_tau_dm'
    elif grouping == 'njets':
        ff_dnn_tau1 = 'ff_dnn_tau1_njets'

    counts_SR_like, bin_edges = np.histogram(df.data.SR_like[var], weights = df.data.SR_like.weight_qcd, bins = bins)
    counts_FF_AR_like, _ = np.histogram(df.data.AR_like_tau1[var], weights = df.data.AR_like_tau1.weight_qcd * df.data.AR_like_tau1[ff_dnn_tau1], bins = bins)

    variance_SR_like, _ = np.histogram(df.data.SR_like[var], weights = df.data.SR_like.weight_qcd**2, bins = bins)
    variance_FF_AR_like, _ = np.histogram(
        df.data.AR_like_tau1[var], 
        weights = (df.data.AR_like_tau1.weight_qcd * df.data.AR_like_tau1[ff_dnn_tau1])**2,
        bins = bins)

    err_SR_like = np.sqrt(variance_SR_like)
    err_FF_AR_like = np.sqrt(variance_FF_AR_like)

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    err_bin = 0.5 * bin_widths


    fig, ax = plt.subplots(
        2,
        1,
        figsize=(11.7, 9.1),
        sharex=True,
        gridspec_kw={
            'height_ratios': [4, 1],
            'hspace': 0.05,
        },
        constrained_layout=True,
    )
    CMS_LABEL(ax)
    #CMS_CATEGORY_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_CHANNEL_TITLE(ax)

    ax[0].errorbar(
        bin_centers,
        counts_SR_like,
        yerr=err_SR_like,
        xerr=err_bin,
        fmt='o',
        color='black',
        label='data(SR-like)',
        markersize=6,
        elinewidth=1.2,
        capsize=0,
    )

    ax[0].stairs(counts_FF_AR_like, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like)', ls = '--', linewidth = 2)

    ax[0].set_ylabel('Events')
    ax[0].legend()
    adjust_ylim_for_legend(ax[0])
    ratio = np.divide(counts_SR_like, counts_FF_AR_like, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like > 0)
    ratio_err_SR_like = np.divide(err_SR_like, counts_FF_AR_like, out=np.zeros_like(err_SR_like), where=counts_FF_AR_like > 0)
    ratio_err_FF_AR_like = np.divide(err_FF_AR_like, counts_FF_AR_like, out=np.zeros_like(err_FF_AR_like), where=counts_FF_AR_like > 0)

    ax[1].errorbar(bin_centers, ratio, xerr = err_bin, yerr = ratio_err_SR_like, fmt='o', color='black', markersize=6, label='ratio')
    ax[1].fill_between(
        bin_centers,
        1 - ratio_err_FF_AR_like,
        1 + ratio_err_FF_AR_like,
        color='gray',
        alpha=0.3,
        step='mid',
        label='Sys. Unc.',
    )
    ax[1].set_ylabel("Data / Model", loc='center')
    ax[1].set_ylim([0.75, 1.25])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=2, frameon=False)
    ax[1].set_xlabel(label)

    return fig, ax

def FF_closure_in_DR_ttbar(
    df,
	var,
	bins,
	label,
	grouping = 'tau_decaymode',
):
    if grouping == 'tau_decaymode':
        ff_dnn_ttbar = 'ff_dnn_ttbar'
    elif grouping == 'njets':
        ff_dnn_ttbar = 'ff_dnn_ttbar_njets'


    counts_SR_like, bin_edges = np.histogram(df.data.SR_like_ttbar[var], weights = df.data.SR_like_ttbar.weight, bins = bins)
    counts_FF_AR_like, _ = np.histogram(df.data.AR_like_ttbar[var], weights = df.data.AR_like_ttbar.weight * df.data.AR_like_ttbar[ff_dnn_ttbar], bins = bins)

    variance_SR_like, _ = np.histogram(df.data.SR_like_ttbar[var], weights = df.data.SR_like_ttbar.weight**2, bins = bins)
    variance_FF_AR_like, _ = np.histogram(
        df.data.AR_like_ttbar[var], 
        weights = (df.data.AR_like_ttbar.weight * df.data.AR_like_ttbar[ff_dnn_ttbar])**2,
        bins = bins)

    err_SR_like = np.sqrt(variance_SR_like)
    err_FF_AR_like = np.sqrt(variance_FF_AR_like)

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    err_bin = 0.5 * bin_widths


    fig, ax = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={
            'height_ratios': [4, 1],
            'hspace': 0.05,
        },
        constrained_layout=True,
    )
    CMS_LABEL(ax)
    #CMS_CATEGORY_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_CHANNEL_TITLE(ax)

    ax[0].errorbar(
        bin_centers,
        counts_SR_like,
        yerr=err_SR_like,
        xerr=err_bin,
        fmt='o',
        color='black',
        label='data(SR-like)',
        markersize=6,
        elinewidth=1.2,
        capsize=0,
    )

    ax[0].stairs(counts_FF_AR_like, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like)', ls = '--', linewidth = 2)

    ax[0].set_ylabel('Events')
    ax[0].legend()
    adjust_ylim_for_legend(ax[0])
    ratio = np.divide(counts_SR_like, counts_FF_AR_like, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like > 0)
    ratio_err_SR_like = np.divide(err_SR_like, counts_FF_AR_like, out=np.zeros_like(err_SR_like), where=counts_FF_AR_like > 0)
    ratio_err_FF_AR_like = np.divide(err_FF_AR_like, counts_FF_AR_like, out=np.zeros_like(err_FF_AR_like), where=counts_FF_AR_like > 0)

    ax[1].errorbar(bin_centers, ratio, xerr = err_bin, yerr = ratio_err_SR_like, fmt='o', color='black', markersize=6, label='ratio')
    ax[1].fill_between(
        bin_centers,
        1 - ratio_err_FF_AR_like,
        1 + ratio_err_FF_AR_like,
        color='gray',
        alpha=0.3,
        step='mid',
        label='Sys. Unc.',
    )
    ax[1].set_ylabel("Data / Model")
    ax[1].set_ylim([0.75, 1.25])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=2, frameon=False)
    ax[1].set_xlabel(label)

    return fig, ax

def FF_closure_in_DR_ttbar_MC(
    df,
	var,
	bins,
	label,
	grouping = 'tau_decaymode',
):
    if grouping == 'tau_decaymode':
        ff_dnn_ttbar = 'ff_dnn_ttbar'
    elif grouping == 'njets':
        ff_dnn_ttbar = 'ff_dnn_ttbar_njets'


    counts_SR_like, bin_edges = np.histogram(df.ttbar.SR_like_ttbar[var], weights = df.ttbar.SR_like_ttbar.weight, bins = bins)
    counts_FF_AR_like, _ = np.histogram(df.ttbar.AR_like_ttbar[var], weights = df.ttbar.AR_like_ttbar.weight * df.ttbar.AR_like_ttbar[ff_dnn_ttbar], bins = bins)

    variance_SR_like, _ = np.histogram(df.ttbar.SR_like_ttbar[var], weights = df.ttbar.SR_like_ttbar.weight**2, bins = bins)
    variance_FF_AR_like, _ = np.histogram(
        df.ttbar.AR_like_ttbar[var], 
        weights = (df.ttbar.AR_like_ttbar.weight * df.ttbar.AR_like_ttbar[ff_dnn_ttbar])**2,
        bins = bins)

    err_SR_like = np.sqrt(variance_SR_like)
    err_FF_AR_like = np.sqrt(variance_FF_AR_like)

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    err_bin = 0.5 * bin_widths


    fig, ax = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={
            'height_ratios': [4, 1],
            'hspace': 0.05,
        },
        constrained_layout=True,
    )
    #CMS_LABEL(ax)
    #CMS_CATEGORY_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_CHANNEL_TITLE(ax)

    ax[0].errorbar(
        bin_centers,
        counts_SR_like,
        yerr=err_SR_like,
        xerr=err_bin,
        fmt='o',
        color='black',
        label='MC_events(SR-like)',
        markersize=6,
        elinewidth=1.2,
        capsize=0,
    )

    ax[0].stairs(counts_FF_AR_like, bin_edges, label = r'$F_\mathrm{F} \cdot $ MC_events(AR-like)', ls = '--', linewidth = 2)

    ax[0].set_ylabel('Events')
    ax[0].legend()
    adjust_ylim_for_legend(ax[0])
    ratio = np.divide(counts_SR_like, counts_FF_AR_like, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like > 0)
    ratio_err_SR_like = np.divide(err_SR_like, counts_FF_AR_like, out=np.zeros_like(err_SR_like), where=counts_FF_AR_like > 0)
    ratio_err_FF_AR_like = np.divide(err_FF_AR_like, counts_FF_AR_like, out=np.zeros_like(err_FF_AR_like), where=counts_FF_AR_like > 0)

    ax[1].errorbar(bin_centers, ratio, xerr = err_bin, yerr = ratio_err_SR_like, fmt='o', color='black', markersize=6, label='ratio')
    ax[1].fill_between(
        bin_centers,
        1 - ratio_err_FF_AR_like,
        1 + ratio_err_FF_AR_like,
        color='gray',
        alpha=0.3,
        step='mid',
        label='Sys. Unc.',
    )
    ax[1].set_ylabel("MC_events / Model")
    ax[1].set_ylim([0.75, 1.25])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=2, frameon=False)
    ax[1].set_xlabel(label)

    return fig, ax

def FF_closure_in_DR_wjets_with_stat_unc(
    df,
	var,
	bins,
	label,
	grouping = 'tau_decaymode',
):
    ff_nominal = df.data.AR_like_wjets.ff_dnn_wjets
    
    ff_mean = df.data.AR_like_wjets.ff_wjets_mean
    ff_std = df.data.AR_like_wjets.ff_wjets_std

    ff_unc = np.sqrt((ff_nominal - ff_mean)**2 + (ff_std/2)**2)
    
    ff_up = ff_nominal + ff_unc
    ff_down = ff_nominal - ff_unc

    ff_mean_p = df.data.AR_like_wjets.ff_wjets_mean_pmask
    ff_std_p = df.data.AR_like_wjets.ff_wjets_std_pmask

    ff_unc_p = np.sqrt((ff_nominal - ff_mean_p)**2 + (ff_std/2)**2)
    ff_unc_p2 = np.sqrt((ff_nominal - ff_mean_p)**2 + (ff_std)**2)

    ff_up_p = ff_nominal + ff_unc_p
    ff_down_p = ff_nominal - ff_unc_p

    ff_up_p2 = ff_nominal + ff_unc_p2
    ff_down_p2 = ff_nominal - ff_unc_p2

    counts_SR_like, bin_edges = np.histogram(df.data.SR_like_wjets[var], weights = df.data.SR_like_wjets.weight_wjets, bins = bins)
    counts_FF_AR_like_nominal, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_nominal, bins = bins)

    counts_FF_AR_like_up, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_up, bins = bins)
    counts_FF_AR_like_down, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_down, bins = bins)

    counts_FF_AR_like_up_p, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_up_p, bins = bins)
    counts_FF_AR_like_down_p, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_down_p, bins = bins)

    counts_FF_AR_like_up_p2, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_up_p2, bins = bins)
    counts_FF_AR_like_down_p2, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_down_p2, bins = bins)

    variance_SR_like, _ = np.histogram(df.data.SR_like_wjets[var], weights = df.data.SR_like_wjets.weight_wjets**2, bins = bins)
    variance_FF_AR_like, _ = np.histogram(
        df.data.AR_like_wjets[var], 
        weights = (df.data.AR_like_wjets.weight_wjets * ff_nominal)**2,
        bins = bins)

    err_SR_like = np.sqrt(variance_SR_like)
    err_FF_AR_like = np.sqrt(variance_FF_AR_like)

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    err_bin = 0.5 * bin_widths

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={
            'height_ratios': [4, 1],
            'hspace': 0.05,
        },
        constrained_layout=True,
    )
    CMS_LABEL(ax)
    #CMS_CATEGORY_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_CHANNEL_TITLE(ax)

    ax[0].errorbar(
        bin_centers,
        counts_SR_like,
        yerr=err_SR_like,
        xerr=err_bin,
        fmt='o',
        color='black',
        label='data(SR-like)',
        markersize=6,
        elinewidth=1.2,
        capsize=0,
    )

    ax[0].stairs(counts_FF_AR_like_up, bin_edges, label = r'$F_\mathrm{F} \cdot $data(AR-like) up, down (ensemble)', color = 'red', ls = '--', linewidth = 2, alpha = 0.7)
    ax[0].stairs(counts_FF_AR_like_down, bin_edges, color = 'red', ls = '--', linewidth = 2, alpha = 0.7)

    ax[0].stairs(counts_FF_AR_like_up_p, bin_edges, label = r'$F_\mathrm{F} \cdot $data(AR-like) up, down (dropout, 1 $\sigma$)', color = 'orange', ls = '--', linewidth = 2, alpha = 0.7)
    ax[0].stairs(counts_FF_AR_like_down_p, bin_edges, color = 'orange', ls = '--', linewidth = 2, alpha = 0.7)

    ax[0].stairs(counts_FF_AR_like_up_p, bin_edges, label = r'$F_\mathrm{F} \cdot $data(AR-like) up, down (dropout, 1 $\sigma$)', color = 'orange', ls = ':', linewidth = 2, alpha = 0.7)
    ax[0].stairs(counts_FF_AR_like_down_p, bin_edges, color = 'orange', ls = ':', linewidth = 2, alpha = 0.7)

    ax[0].stairs(counts_FF_AR_like_nominal, bin_edges, label = r'$F_\mathrm{F} \cdot $data(AR-like) nominal', color = 'blue', ls = '-', linewidth = 2, alpha = 1.0)


    ax[0].set_ylabel('Events')
    ax[0].legend(loc= 'upper right', fontsize = 'x-small')
    adjust_ylim_for_legend(ax[0])

    ratio = np.divide(counts_SR_like, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
    ratio_up = np.divide(counts_FF_AR_like_up, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
    ratio_down = np.divide(counts_FF_AR_like_down, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
 
    ratio_up_p = np.divide(counts_FF_AR_like_up_p, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
    ratio_down_p = np.divide(counts_FF_AR_like_down_p, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
 
    ratio_up_p2 = np.divide(counts_FF_AR_like_up_p2, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
    ratio_down_p2 = np.divide(counts_FF_AR_like_down_p2, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
 
    ratio_err_SR_like = np.divide(err_SR_like, counts_FF_AR_like_nominal, out=np.zeros_like(err_SR_like), where=counts_FF_AR_like_nominal > 0)
    ratio_err_FF_AR_like = np.divide(err_FF_AR_like, counts_FF_AR_like_nominal, out=np.zeros_like(err_FF_AR_like), where=counts_FF_AR_like_nominal > 0)

    ax[1].errorbar(bin_centers, ratio, xerr = err_bin, yerr = ratio_err_SR_like, fmt='o', color='black', markersize=6, label='ratio')
    ax[1].fill_between(
        bin_centers,
        1 - ratio_err_FF_AR_like,
        1 + ratio_err_FF_AR_like,
        color='gray',
        alpha=0.3,
        step='mid',
        label='Sys. Unc.',
    )
    ax[1].stairs(ratio_up, bin_edges, label = r'up, down (ensemble, 1 $\sigma$)', color = 'red', ls = '--', linewidth = 2, alpha = 0.7)
    ax[1].stairs(ratio_down, bin_edges, color = 'red', ls = '--', linewidth = 2, alpha = 0.7)
    ax[1].stairs(ratio_up_p, bin_edges, label = r'up, down (dropout, 1 $\sigma$)', color = 'orange', ls = '--', linewidth = 2, alpha = 0.7)
    ax[1].stairs(ratio_down_p, bin_edges, color = 'orange', ls = '--', linewidth = 2, alpha = 0.7)
    ax[1].stairs(ratio_up_p2, bin_edges, label = r'up, down (dropout, 2 $\sigma$)', color = 'orange', ls = ':', linewidth = 2, alpha = 0.7)
    ax[1].stairs(ratio_down_p2, bin_edges, color = 'orange', ls = '--', linewidth = 2, alpha = 0.7)


    ax[1].set_ylabel("Data / Model")
    ax[1].set_ylim([0.75, 1.25])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=3, frameon=False)
    ax[1].set_xlabel(label)

    return fig, ax

def FF_closure_in_DR_qcd_with_stat_unc(
    df,
	var,
	bins,
	label,
	grouping = 'tau_decaymode',
):
    ff_nominal = df.data.AR_like_qcd.ff_dnn_qcd
    
    ff_mean = df.data.AR_like_qcd.ff_qcd_mean
    ff_std = df.data.AR_like_qcd.ff_qcd_std

    ff_unc = np.sqrt((ff_nominal - ff_mean)**2 + (ff_std/2)**2)
    
    ff_up = ff_nominal + ff_unc
    ff_down = ff_nominal - ff_unc

    ff_mean_p = df.data.AR_like_qcd.ff_qcd_mean_pmask
    ff_std_p = df.data.AR_like_qcd.ff_qcd_std_pmask

    ff_unc_p = np.sqrt((ff_nominal - ff_mean_p)**2 + (ff_std/2)**2)
    ff_unc_p2 = np.sqrt((ff_nominal - ff_mean_p)**2 + (ff_std)**2)

    ff_up_p = ff_nominal + ff_unc_p
    ff_down_p = ff_nominal - ff_unc_p

    ff_up_p2 = ff_nominal + ff_unc_p2
    ff_down_p2 = ff_nominal - ff_unc_p2

    counts_SR_like, bin_edges = np.histogram(df.data.SR_like_qcd[var], weights = df.data.SR_like_qcd.weight_qcd, bins = bins)
    counts_FF_AR_like_nominal, _ = np.histogram(df.data.AR_like_qcd[var], weights = df.data.AR_like_qcd.weight_qcd * ff_nominal, bins = bins)

    counts_FF_AR_like_up, _ = np.histogram(df.data.AR_like_qcd[var], weights = df.data.AR_like_qcd.weight_qcd * ff_up, bins = bins)
    counts_FF_AR_like_down, _ = np.histogram(df.data.AR_like_qcd[var], weights = df.data.AR_like_qcd.weight_qcd * ff_down, bins = bins)

    counts_FF_AR_like_up_p, _ = np.histogram(df.data.AR_like_qcd[var], weights = df.data.AR_like_qcd.weight_qcd * ff_up_p, bins = bins)
    counts_FF_AR_like_down_p, _ = np.histogram(df.data.AR_like_qcd[var], weights = df.data.AR_like_qcd.weight_qcd * ff_down_p, bins = bins)

    counts_FF_AR_like_up_p2, _ = np.histogram(df.data.AR_like_qcd[var], weights = df.data.AR_like_qcd.weight_qcd * ff_up_p2, bins = bins)
    counts_FF_AR_like_down_p2, _ = np.histogram(df.data.AR_like_qcd[var], weights = df.data.AR_like_qcd.weight_qcd * ff_down_p2, bins = bins)

    variance_SR_like, _ = np.histogram(df.data.SR_like_qcd[var], weights = df.data.SR_like_qcd.weight_qcd**2, bins = bins)
    variance_FF_AR_like, _ = np.histogram(
        df.data.AR_like_qcd[var], 
        weights = (df.data.AR_like_qcd.weight_qcd * ff_nominal)**2,
        bins = bins)

    err_SR_like = np.sqrt(variance_SR_like)
    err_FF_AR_like = np.sqrt(variance_FF_AR_like)

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    err_bin = 0.5 * bin_widths

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={
            'height_ratios': [4, 1],
            'hspace': 0.05,
        },
        constrained_layout=True,
    )
    CMS_LABEL(ax)
    #CMS_CATEGORY_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_CHANNEL_TITLE(ax)

    ax[0].errorbar(
        bin_centers,
        counts_SR_like,
        yerr=err_SR_like,
        xerr=err_bin,
        fmt='o',
        color='black',
        label='data(SR-like)',
        markersize=6,
        elinewidth=1.2,
        capsize=0,
    )

    ax[0].stairs(counts_FF_AR_like_up, bin_edges, label = r'$F_\mathrm{F} \cdot $data(AR-like) up, down (ensemble)', color = 'red', ls = '--', linewidth = 2, alpha = 0.7)
    ax[0].stairs(counts_FF_AR_like_down, bin_edges, color = 'red', ls = '--', linewidth = 2, alpha = 0.7)

    ax[0].stairs(counts_FF_AR_like_up_p, bin_edges, label = r'$F_\mathrm{F} \cdot $data(AR-like) up, down (dropout, 1 $\sigma$)', color = 'orange', ls = '--', linewidth = 2, alpha = 0.7)
    ax[0].stairs(counts_FF_AR_like_down_p, bin_edges, color = 'orange', ls = '--', linewidth = 2, alpha = 0.7)

    ax[0].stairs(counts_FF_AR_like_up_p, bin_edges, label = r'$F_\mathrm{F} \cdot $data(AR-like) up, down (dropout, 1 $\sigma$)', color = 'orange', ls = ':', linewidth = 2, alpha = 0.7)
    ax[0].stairs(counts_FF_AR_like_down_p, bin_edges, color = 'orange', ls = ':', linewidth = 2, alpha = 0.7)

    ax[0].stairs(counts_FF_AR_like_nominal, bin_edges, label = r'$F_\mathrm{F} \cdot $data(AR-like) nominal', color = 'blue', ls = '-', linewidth = 2, alpha = 1.0)


    ax[0].set_ylabel('Events')
    ax[0].legend(loc= 'upper right', fontsize = 'x-small')
    adjust_ylim_for_legend(ax[0])

    ratio = np.divide(counts_SR_like, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
    ratio_up = np.divide(counts_FF_AR_like_up, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
    ratio_down = np.divide(counts_FF_AR_like_down, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
 
    ratio_up_p = np.divide(counts_FF_AR_like_up_p, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
    ratio_down_p = np.divide(counts_FF_AR_like_down_p, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
 
    ratio_up_p2 = np.divide(counts_FF_AR_like_up_p2, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
    ratio_down_p2 = np.divide(counts_FF_AR_like_down_p2, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
 
    ratio_err_SR_like = np.divide(err_SR_like, counts_FF_AR_like_nominal, out=np.zeros_like(err_SR_like), where=counts_FF_AR_like_nominal > 0)
    ratio_err_FF_AR_like = np.divide(err_FF_AR_like, counts_FF_AR_like_nominal, out=np.zeros_like(err_FF_AR_like), where=counts_FF_AR_like_nominal > 0)

    ax[1].errorbar(bin_centers, ratio, xerr = err_bin, yerr = ratio_err_SR_like, fmt='o', color='black', markersize=6, label='ratio')
    ax[1].fill_between(
        bin_centers,
        1 - ratio_err_FF_AR_like,
        1 + ratio_err_FF_AR_like,
        color='gray',
        alpha=0.3,
        step='mid',
        label='Sys. Unc.',
    )
    ax[1].stairs(ratio_up, bin_edges, label = r'up, down (ensemble, 1 $\sigma$)', color = 'red', ls = '--', linewidth = 2, alpha = 0.7)
    ax[1].stairs(ratio_down, bin_edges, color = 'red', ls = '--', linewidth = 2, alpha = 0.7)
    ax[1].stairs(ratio_up_p, bin_edges, label = r'up, down (dropout, 1 $\sigma$)', color = 'orange', ls = '--', linewidth = 2, alpha = 0.7)
    ax[1].stairs(ratio_down_p, bin_edges, color = 'orange', ls = '--', linewidth = 2, alpha = 0.7)
    ax[1].stairs(ratio_up_p2, bin_edges, label = r'up, down (dropout, 2 $\sigma$)', color = 'orange', ls = ':', linewidth = 2, alpha = 0.7)
    ax[1].stairs(ratio_down_p2, bin_edges, color = 'orange', ls = '--', linewidth = 2, alpha = 0.7)


    ax[1].set_ylabel("Data / Model")
    ax[1].set_ylim([0.75, 1.25])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=3, frameon=False)
    ax[1].set_xlabel(label)

    return fig, ax






def FF_closure_in_DR_wjets_with_stat_unc_p_mask_2sigma(
    df,
	var,
	bins,
	label,
	grouping = 'tau_decaymode',
):
    ff_nominal = df.data.AR_like_wjets.ff_dnn_wjets
    ff_unc = df.data.AR_like_wjets.ff_wjets_unc_pmask
    ff_up = ff_nominal + 2*ff_unc
    ff_down = ff_nominal - 2*ff_unc

    counts_SR_like, bin_edges = np.histogram(df.data.SR_like_wjets[var], weights = df.data.SR_like_wjets.weight_wjets, bins = bins)
    counts_FF_AR_like_nominal, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_nominal, bins = bins)
    counts_FF_AR_like_up, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_up, bins = bins)
    counts_FF_AR_like_down, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_down, bins = bins)
    variance_SR_like, _ = np.histogram(df.data.SR_like_wjets[var], weights = df.data.SR_like_wjets.weight_wjets**2, bins = bins)
    variance_FF_AR_like, _ = np.histogram(
        df.data.AR_like_wjets[var], 
        weights = (df.data.AR_like_wjets.weight_wjets * ff_nominal)**2,
        bins = bins)

    err_SR_like = np.sqrt(variance_SR_like)
    err_FF_AR_like = np.sqrt(variance_FF_AR_like)

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    err_bin = 0.5 * bin_widths


    fig, ax = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={
            'height_ratios': [4, 1],
            'hspace': 0.05,
        },
        constrained_layout=True,
    )
    CMS_LABEL(ax)
    #CMS_CATEGORY_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_CHANNEL_TITLE(ax)

    ax[0].errorbar(
        bin_centers,
        counts_SR_like,
        yerr=err_SR_like,
        xerr=err_bin,
        fmt='o',
        color='black',
        label='data(SR-like)',
        markersize=6,
        elinewidth=1.2,
        capsize=0,
    )

    ax[0].stairs(counts_FF_AR_like_nominal, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like) nominal', color = 'blue', ls = '-', linewidth = 2, alpha = 0.7)
    ax[0].stairs(counts_FF_AR_like_up, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like) up', color = 'red', ls = '--', linewidth = 2, alpha = 0.4)
    ax[0].stairs(counts_FF_AR_like_down, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like) dowm', color = 'orange', ls = '--', linewidth = 2, alpha = 0.4)

    ax[0].set_ylabel('Events')
    ax[0].legend()
    adjust_ylim_for_legend(ax[0])

    ratio = np.divide(counts_SR_like, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
    ratio_up = np.divide(counts_FF_AR_like_up, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
    ratio_down = np.divide(counts_FF_AR_like_down, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)

   
    ratio_err_SR_like = np.divide(err_SR_like, counts_FF_AR_like_nominal, out=np.zeros_like(err_SR_like), where=counts_FF_AR_like_nominal > 0)
    ratio_err_FF_AR_like = np.divide(err_FF_AR_like, counts_FF_AR_like_nominal, out=np.zeros_like(err_FF_AR_like), where=counts_FF_AR_like_nominal > 0)

    ax[1].errorbar(bin_centers, ratio, xerr = err_bin, yerr = ratio_err_SR_like, fmt='o', color='black', markersize=6, label='ratio')
    ax[1].fill_between(
        bin_centers,
        1 - ratio_err_FF_AR_like,
        1 + ratio_err_FF_AR_like,
        color='gray',
        alpha=0.3,
        step='mid',
        label='Sys. Unc.',
    )
    ax[1].stairs(ratio_up, bin_edges, label = r'up', color = 'red', ls = '--', linewidth = 2, alpha = 0.3)
    ax[1].stairs(ratio_down, bin_edges, label = r'down', color = 'orange', ls = '--', linewidth = 2, alpha = 0.3)
    ax[1].set_ylabel("Data / Model")
    ax[1].set_ylim([0.75, 1.25])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=2, frameon=False)
    ax[1].set_xlabel(label)

    return fig, ax



def FF_closure_in_DR_wjets_with_stat_unc_ensemble(
    df,
	var,
	bins,
	label,
):
    ff_nominal = df.data.AR_like_wjets.ff_wjets_nominal_ensemble
    ff_up = df.data.AR_like_wjets.ff_wjets_up_ensemble
    ff_down = df.data.AR_like_wjets.ff_wjets_down_ensemble

    counts_SR_like, bin_edges = np.histogram(df.data.SR_like_wjets[var], weights = df.data.SR_like_wjets.weight_wjets, bins = bins)
    counts_FF_AR_like_nominal, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_nominal, bins = bins)
    counts_FF_AR_like_up, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_up, bins = bins)
    counts_FF_AR_like_down, _ = np.histogram(df.data.AR_like_wjets[var], weights = df.data.AR_like_wjets.weight_wjets * ff_down, bins = bins)
    variance_SR_like, _ = np.histogram(df.data.SR_like_wjets[var], weights = df.data.SR_like_wjets.weight_wjets**2, bins = bins)
    variance_FF_AR_like, _ = np.histogram(
        df.data.AR_like_wjets[var], 
        weights = (df.data.AR_like_wjets.weight_wjets * ff_nominal)**2,
        bins = bins)

    err_SR_like = np.sqrt(variance_SR_like)
    err_FF_AR_like = np.sqrt(variance_FF_AR_like)

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    err_bin = 0.5 * bin_widths


    fig, ax = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={
            'height_ratios': [4, 1],
            'hspace': 0.05,
        },
        constrained_layout=True,
    )
    CMS_LABEL(ax)
    #CMS_CATEGORY_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_CHANNEL_TITLE(ax)

    ax[0].errorbar(
        bin_centers,
        counts_SR_like,
        yerr=err_SR_like,
        xerr=err_bin,
        fmt='o',
        color='black',
        label='data(SR-like)',
        markersize=6,
        elinewidth=1.2,
        capsize=0,
    )

    ax[0].stairs(counts_FF_AR_like_nominal, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like) nominal', color = 'blue', ls = '-', linewidth = 2, alpha = 0.7)
    ax[0].stairs(counts_FF_AR_like_up, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like) up', color = 'red', ls = '--', linewidth = 2, alpha = 0.4)
    ax[0].stairs(counts_FF_AR_like_down, bin_edges, label = r'$F_\mathrm{F} \cdot $ data(AR-like) dowm', color = 'orange', ls = '--', linewidth = 2, alpha = 0.4)

    ax[0].set_ylabel('Events')
    ax[0].legend()
    adjust_ylim_for_legend(ax[0])

    ratio = np.divide(counts_SR_like, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
    ratio_up = np.divide(counts_FF_AR_like_up, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)
    ratio_down = np.divide(counts_FF_AR_like_down, counts_FF_AR_like_nominal, out=np.zeros_like(counts_SR_like, dtype=float), where=counts_FF_AR_like_nominal > 0)


    ratio_err_SR_like = np.divide(err_SR_like, counts_FF_AR_like_nominal, out=np.zeros_like(err_SR_like), where=counts_FF_AR_like_nominal > 0)
    ratio_err_FF_AR_like = np.divide(err_FF_AR_like, counts_FF_AR_like_nominal, out=np.zeros_like(err_FF_AR_like), where=counts_FF_AR_like_nominal > 0)

    ax[1].errorbar(bin_centers, ratio, xerr = err_bin, yerr = ratio_err_SR_like, fmt='o', color='black', markersize=6, label='ratio')
    ax[1].fill_between(
        bin_centers,
        1 - ratio_err_FF_AR_like,
        1 + ratio_err_FF_AR_like,
        color='gray',
        alpha=0.3,
        step='mid',
        label='Sys. Unc.',
    )
    ax[1].stairs(ratio_up, bin_edges, label = r'up', color = 'red', ls = '--', linewidth = 2, alpha = 0.3)
    ax[1].stairs(ratio_down, bin_edges, label = r'down', color = 'orange', ls = '--', linewidth = 2, alpha = 0.3)
    ax[1].set_ylabel("Data / Model")
    ax[1].set_ylim([0.75, 1.25])
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].tick_params(direction='in', top=True, right=True)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, ncol=2, frameon=False)
    ax[1].set_xlabel(label)

    return fig, ax


def _grouping_masks(frame, grouping):
    if grouping == 'tau_decaymode':
        grouping = 'tau_decaymode_2'

    if grouping == 'tau_decaymode_2':
        return [
            (frame.tau_decaymode_2 == 0, r't_dm $=$ 0'),
            (frame.tau_decaymode_2 == 1, r't_dm $=$ 1'),
            (frame.tau_decaymode_2 == 10, r't_dm $=$ 10'),
            (frame.tau_decaymode_2 == 11, r't_dm $=$ 11'),
        ]

    if grouping == 'njets':
        return [
            (frame.njets == 0, r'njets $=$ 0'),
            (frame.njets == 1, r'njets $=$ 1'),
            (frame.njets >= 2, r'njets $\geq$ 2'),
        ]

    if grouping == 'tau_decaymode_1':
        return [
            (frame.tau_decaymode_1 == 0, r't_dm $=$ 0'),
            (frame.tau_decaymode_1 == 1, r't_dm $=$ 1'),
            (frame.tau_decaymode_1 == 10, r't_dm $=$ 10'),
            (frame.tau_decaymode_1 == 11, r't_dm $=$ 11'),
        ]

    raise ValueError(f'Unsupported grouping: {grouping}')



def plot_fake_factors_grouped_c(df, category_title, grouping='tau_decaymode', squeeze_upper_bound = '0'):
    if grouping == 'tau_decaymode':
        ff_wjets = f'ff_dnn_wjets_{squeeze_upper_bound}'
        ff_qcd = f'ff_dnn_qcd_{squeeze_upper_bound}'
        ff_ttbar = f'ff_dnn_ttbar_{squeeze_upper_bound}'
    elif grouping == 'njets':
        ff_wjets = 'ff_dnn_wjets_njets'
        ff_qcd = 'ff_dnn_qcd_njets'
        ff_ttbar = 'ff_dnn_ttbar_njets'
    else:
        raise ValueError(f'Unsupported grouping: {grouping}')

    bins_wjets = np.linspace(0, 1, 50)
    bins_qcd = np.linspace(0, 0.5, 50)
    bins_ttbar = np.linspace(0, 1, 50)

    frame = df.data.AR

    fig, ax = plt.subplots(3, 1, figsize=(10, 7))

    ax[0].hist(frame[ff_wjets], bins=bins_wjets, histtype='step', linewidth=2, label='Wjets: incl')
    for mask, mask_label in _grouping_masks(frame, grouping):
        ax[0].hist(frame[ff_wjets][mask], bins=bins_wjets, histtype='step', ls='--', label=f'{mask_label}')
    ax[0].set_ylabel('Events')
    ax[0].legend(prop={'size': 9}, loc = 'upper right')


    CMS_CHANNEL_TITLE([ax[0]])
    CMS_LUMI_TITLE([ax[0]])
    CMS_LABEL([ax[0]])
    CMS_CATEGORY_TITLE([ax[0]], title=category_title)

    ax[1].set_ylabel('Events')
    ax[1].hist(frame[ff_qcd], bins=bins_qcd, histtype='step', linewidth=2, label='QCD: incl')
    for mask, mask_label in _grouping_masks(frame, grouping):
        ax[1].hist(frame[ff_qcd][mask], bins=bins_qcd, histtype='step', ls='--', label=f'{mask_label}')
    ax[1].legend(prop={'size': 9})

    ax[2].set_ylabel('Events')
    ax[2].hist(frame[ff_ttbar], bins=bins_ttbar, histtype='step', linewidth=2, label='ttbar: incl')
    for mask, mask_label in _grouping_masks(frame, grouping):
        ax[2].hist(frame[ff_ttbar][mask], bins=bins_ttbar, histtype='step', ls='--', label=f'{mask_label}')
    ax[2].set_xlabel(r'$F_{\mathrm{F}}$ value')
    ax[2].legend(prop={'size': 9})



    return fig, ax


def plot_fake_factors_grouped(df, category_title, grouping='tau_decaymode', clipped = True):
    hep.style.use(hep.style.CMS)

    if clipped:
        bins_tau1 = np.linspace(0, 1.0, 50)
        bins_tau2 = np.linspace(0, 1.0, 50)

        if grouping == 'tau_decaymode':
            ff_tau1 = 'ff_dnn_tau1_tau_dm'
            ff_tau2 = 'ff_dnn_tau2_tau_dm'
            grouping = ['tau_decaymode_1', 'tau_decaymode_2']
        elif grouping == 'njets':
            ff_tau1 = 'ff_dnn_tau1_njets'
            ff_tau2 = 'ff_dnn_tau2_njets'
        else:
            raise ValueError(f'Unsupported grouping: {grouping}')
    else:
        bins_tau1 = np.linspace(0, 2., 50)
        bins_tau2 = np.linspace(0, 2., 50)

        if grouping == 'tau_decaymode':
            ff_tau1 = 'ff_unclipped_dnn_tau1_tau_dm'
            ff_tau2 = 'ff_unclipped_dnn_tau2_tau_dm'
            grouping = ['tau_decaymode_1', 'tau_decaymode_2']
        elif grouping == 'njets':
            ff_tau1 = 'ff_unclipped_dnn_tau1_njets'
            ff_tau2 = 'ff_unclipped_dnn_tau2_njets'
        else:
            raise ValueError(f'Unsupported grouping: {grouping}')


    frame_tau1 = df.data.AR_tau1
    frame_tau2 = df.data.AR_tau2

    if isinstance(grouping, list):
        group_mask_tau1 = _grouping_masks(frame_tau1, grouping[0])
        group_mask_tau2 = _grouping_masks(frame_tau2, grouping[1])
    else:
        group_mask_tau1 = _grouping_masks(frame_tau1, grouping)
        group_mask_tau2 = _grouping_masks(frame_tau2, grouping)

    fig, ax = plt.subplots(2, 1, figsize=(11.7, 9.1))

    
    for x in frame_tau1[ff_tau1]:
        if x > 3.0: print(x)


    n1 = ax[0].hist(frame_tau1[ff_tau1], bins=bins_tau1, histtype='step', linewidth=2, label=r'Leading $\tau_h$: incl')
    for mask, mask_label in group_mask_tau1:
        ax[0].hist(frame_tau1[ff_tau1][mask], bins=bins_tau1, histtype='step', ls='--', label=f'{mask_label}')
    ax[0].set_ylabel('Events')
    ax[0].legend(loc = 'center left', prop={'size': 15})
    ax[0].set_ylim(top=1.2 * np.max(n1[0]))


    CMS_CHANNEL_TITLE([ax[0]])
    CMS_LUMI_TITLE([ax[0]])
    CMS_LABEL([ax[0]])
    CMS_CATEGORY_TITLE([ax[0]], title=category_title)

    ax[1].set_ylabel('Events')
    n2 = ax[1].hist(frame_tau2[ff_tau2], bins=bins_tau2, histtype='step', linewidth=2, label=r'Trailing $\tau_h$: incl')
    for mask, mask_label in group_mask_tau2:
        ax[1].hist(frame_tau2[ff_tau2][mask], bins=bins_tau2, histtype='step', ls='--', label=f'{mask_label}')
    ax[1].set_xlabel(r'$F_{\mathrm{F}}$ value')
    ax[1].legend(loc = 'upper left', prop={'size': 15})

    return fig, ax


def plot_fake_factors_grouped_combTaus(df, category_title, grouping='tau_decaymode'):
    hep.style.use(hep.style.CMS)

    if grouping == 'tau_decaymode':
        ff = 'ff_dnn_tau_dm'
        grouping = ['tau_decaymode_1', 'tau_decaymode_2']
    elif grouping == 'njets':
        ff = 'ff_dnn_njets'
    else:
        raise ValueError(f'Unsupported grouping: {grouping}')

    bins_tau1 = np.linspace(0, 0.5, 50)
    bins_tau2 = np.linspace(0, 0.5, 50)

    frame_tau1 = df.data.AR_tau1
    frame_tau2 = df.data.AR_tau2

    if isinstance(grouping, list):
        group_mask_tau1 = _grouping_masks(frame_tau1, grouping[0])
        group_mask_tau2 = _grouping_masks(frame_tau2, grouping[1])
    else:
        group_mask_tau1 = _grouping_masks(frame_tau1, grouping)
        group_mask_tau2 = _grouping_masks(frame_tau2, grouping)

    n1, binedges = np.histogram(frame_tau1[ff], bins=bins_tau1)
    n2, _ = np.histogram(frame_tau2[ff], bins=bins_tau2)

    n = n1 + n2

    n1_split = []
    n2_split = []

    for mask, mask_label in group_mask_tau1:
        h,_ = np.histogram(frame_tau1[ff][mask], bins=bins_tau1)
        n1_split.append(h)

    for mask, mask_label in group_mask_tau2:
        h,_ = np.histogram(frame_tau2[ff][mask], bins=bins_tau2)
        n2_split.append(h)

    n_split = []
    for i in range(len(n1_split)):
        n_split.append(n1_split[i] + n2_split[i])
      
    fig, ax = plt.subplots(1, 1, figsize=(11.7, 9.1))
    CMS_CHANNEL_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_LABEL(ax)
    CMS_CATEGORY_TITLE(ax, title=category_title)

    ax.stairs(n, binedges, linewidth=2, label=r'Combined $\tau_h$: incl')

    for x, (_, mask_label) in zip(n_split, group_mask_tau1):
        ax.stairs(x, binedges, ls='--', label=f'{mask_label}')

    ax.set_ylabel('Events')
    ax.set_xlabel(r'$F_{\mathrm{F}}$ value')
    ax.set_ylabel('Events')
    ax.legend(loc = 'upper right', prop={'size': 20})

    return fig, ax


def plot_fake_factors_in_dr_grouped(df, category_title, grouping='tau_decaymode'):
    hep.style.use(hep.style.CMS)

    if grouping == 'tau_decaymode':
        ff_tau1 = 'ff_dnn_tau1_tau_dm'
        ff_tau2 = 'ff_dnn_tau2_tau_dm'
        grouping = ['tau_decaymode_1', 'tau_decaymode_2']
    elif grouping == 'njets':
        ff_tau1 = 'ff_dnn_tau1_njets'
        ff_tau2 = 'ff_dnn_tau2_njets'
    else:
        raise ValueError(f'Unsupported grouping: {grouping}')

    bins_tau1 = np.linspace(0, 1.0, 50)
    bins_tau2 = np.linspace(0, 1.0, 50)

    frame_tau1 = df.data.AR_like_tau1
    frame_tau2 = df.data.AR_like_tau2

    if isinstance(grouping, list):
        group_mask_tau1 = _grouping_masks(frame_tau1, grouping[0])
        group_mask_tau2 = _grouping_masks(frame_tau2, grouping[1])
    else:
        group_mask_tau1 = _grouping_masks(frame_tau1, grouping)
        group_mask_tau2 = _grouping_masks(frame_tau2, grouping)

    fig, ax = plt.subplots(2, 1, figsize=(11.7, 9.1))
    n1 = ax[0].hist(frame_tau1[ff_tau1], bins=bins_tau1, histtype='step', linewidth=2, label=r'Leading $\tau_h$: incl')
    for mask, mask_label in group_mask_tau1:
        ax[0].hist(frame_tau1[ff_tau1][mask], bins=bins_tau1, histtype='step', ls='--', label=f'{mask_label}')
    ax[0].set_ylabel('Events')
    ax[0].legend(loc = 'center left', prop={'size': 15})
    ax[0].set_ylim(top=1.2 * np.max(n1[0]))

    CMS_CHANNEL_TITLE([ax[0]])
    CMS_LUMI_TITLE([ax[0]])
    CMS_LABEL([ax[0]])
    CMS_CATEGORY_TITLE([ax[0]], title=category_title)

    ax[1].set_ylabel('Events')
    ax[1].hist(frame_tau2[ff_tau2], bins=bins_tau2, histtype='step', linewidth=2, label=r'Trailing $\tau_h$: incl')
    for mask, mask_label in group_mask_tau2:
        ax[1].hist(frame_tau2[ff_tau2][mask], bins=bins_tau2, histtype='step', ls='--', label=f'{mask_label}')
    ax[1].legend(loc = 'upper left', prop={'size': 15})

    return fig, ax


def plot_fake_factors_in_dr_grouped_c(df, category_title, squeeze_upper_bound, grouping='tau_decaymode'):
    
    if grouping == 'tau_decaymode':
        ff_wjets = f'ff_dnn_wjets_{squeeze_upper_bound}'
        ff_qcd = f'ff_dnn_qcd_{squeeze_upper_bound}'
        ff_ttbar = f'ff_dnn_ttbar_{squeeze_upper_bound}'
    else:
        raise ValueError(f'Unsupported grouping: {grouping}')

    bins_wjets = np.linspace(0, 1, 50)
    bins_qcd = np.linspace(0, 1, 50)
    bins_ttbar = np.linspace(0, 1, 50)

    frame_wjets = df.data.AR_like_wjets
    frame_qcd = df.data.AR_like_qcd
    frame_ttbar = df.data.AR_like_ttbar

    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    ax[0].hist(frame_wjets[ff_wjets], bins=bins_wjets, histtype='step', linewidth=2, label='Wjets: incl')
    for mask, mask_label in _grouping_masks(frame_wjets, grouping):
        ax[0].hist(frame_wjets[ff_wjets][mask], bins=bins_wjets, histtype='step', ls='--', label=f'{mask_label}')
    ax[0].set_ylabel('Events')
    ax[0].legend(prop={'size': 9}, loc = 'upper right')

    CMS_CHANNEL_TITLE([ax[0]])
    CMS_LUMI_TITLE([ax[0]])
    CMS_LABEL([ax[0]])
    CMS_CATEGORY_TITLE([ax[0]], title=category_title)

    ax[1].set_ylabel('Events')
    ax[1].hist(frame_qcd[ff_qcd], bins=bins_qcd, histtype='step', linewidth=2, label='QCD: incl')
    for mask, mask_label in _grouping_masks(frame_qcd, grouping):
        ax[1].hist(frame_qcd[ff_qcd][mask], bins=bins_qcd, histtype='step', ls='--', label=f'{mask_label}')
    ax[1].legend(prop={'size': 9})

    ax[2].set_ylabel('Events')
    ax[2].hist(frame_ttbar[ff_ttbar], bins=bins_ttbar, histtype='step', linewidth=2, label='ttbar: incl')
    for mask, mask_label in _grouping_masks(frame_ttbar, grouping):
        ax[2].hist(frame_ttbar[ff_ttbar][mask], bins=bins_ttbar, histtype='step', ls='--', label=f'{mask_label}')
    ax[2].set_xlabel(r'$F_{\mathrm{F}}$ value')
    ax[2].legend(prop={'size': 9})

    return fig, ax

def plot_NN_output_FF(
        NN_output_SR_like,
        NN_output_AR_like,
        FF,
        process = 'Wjets',
        ):




    bins = np.linspace(0, 1, 50)

    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    ax[0].hist(NN_output_SR_like, bins=bins, histtype='step', linewidth=2, label='SR-like')
    ax[0].hist(NN_output_AR_like, bins=bins, histtype='step', linewidth=2, label='AR-like')
    
    ax[0].set_ylabel('Events')
    ax[0].set_ylabel('NN output')
    ax[0].legend(prop={'size': 9}, loc = 'upper right')
    ax[0].set_ylim(0, 20000)

    CMS_CHANNEL_TITLE([ax[0]])
    CMS_LUMI_TITLE([ax[0]])
    CMS_LABEL([ax[0]])
    CMS_CATEGORY_TITLE([ax[0]], title=process)

    ax[1].set_ylabel('Events')
    ax[1].hist(FF, bins=bins, histtype='step', linewidth=2)
    ax[1].set_xlabel('fake factors')
    return fig, ax



def estimate_jet_fakes(
	df,
	bins,
	var,
	ff_var_tau1,
    ff_var_tau2
):
    counts_tau1 = {}
    counts_tau2 = {}
    variance_tau1 = {}
    variance_tau2 = {}

    list_processes = ['data', 'diboson', 'DYjets', 'ST', 'embedding', 'ttbar', 'wjets']
    for proc in list_processes:
        counts_tau1[proc], _ = np.histogram(df[proc].AR_tau1[var], weights = df[proc].AR_tau1.weight * df[proc].AR_tau1[ff_var_tau1], bins = bins)
        variance_tau1[proc], _ = np.histogram(df[proc].AR_tau1[var], weights = (df[proc].AR_tau1.weight * df[proc].AR_tau1[ff_var_tau1])**2, bins = bins)
        
        counts_tau2[proc], _ = np.histogram(df[proc].AR_tau2[var], weights = df[proc].AR_tau2.weight * df[proc].AR_tau2[ff_var_tau2], bins = bins)
        variance_tau2[proc], _ = np.histogram(df[proc].AR_tau2[var], weights = (df[proc].AR_tau2.weight * df[proc].AR_tau2[ff_var_tau2])**2, bins = bins)

    jet_fakes_tau1 = counts_tau1['data'] - counts_tau1['diboson'] - counts_tau1['DYjets'] - counts_tau1['ST'] - counts_tau1['embedding'] - counts_tau1['ttbar'] - counts_tau1['wjets']
    var_jet_fakes_tau1 = variance_tau1['data'] + variance_tau1['diboson'] + variance_tau1['DYjets'] + variance_tau1['ST'] + variance_tau1['embedding'] + variance_tau1['ttbar'] + variance_tau1['wjets']

    jet_fakes_tau2 = counts_tau2['data'] - counts_tau2['diboson'] - counts_tau2['DYjets'] - counts_tau2['ST'] - counts_tau2['embedding'] - counts_tau2['ttbar'] - counts_tau2['wjets']
    var_jet_fakes_tau2 = variance_tau2['data'] + variance_tau2['diboson'] + variance_tau2['DYjets'] + variance_tau2['ST'] + variance_tau2['embedding'] + variance_tau2['ttbar'] + variance_tau2['wjets']

    jet_fakes = 0.5 * (jet_fakes_tau1 + jet_fakes_tau2)
    var_jet_fakes  = var_jet_fakes_tau1 + var_jet_fakes_tau2

    return jet_fakes, var_jet_fakes


def estimate_jet_fakes_incl(
	df,
	bins,
	var,
	ff_var
):
    counts = {}
    variance = {}

    list_processes = ['data', 'diboson', 'DYjets', 'ST', 'embedding', 'ttbar', 'wjets']
    for proc in list_processes:
        counts[proc], _ = np.histogram(df[proc].AR[var], weights = df[proc].AR.weight * df[proc].AR[ff_var], bins = bins)
        variance[proc], _ = np.histogram(df[proc].AR[var], weights = (df[proc].AR.weight * df[proc].AR[ff_var])**2, bins = bins)

    jet_fakes = counts['data'] - counts['diboson'] - counts['DYjets'] - counts['ST'] - counts['embedding'] - counts['ttbar'] - counts['wjets']
    var_jet_fakes = variance['data'] + variance['diboson'] + variance['DYjets'] + variance['ST'] + variance['embedding'] + variance['ttbar'] + variance['wjets']

    #jet_fakes = 0.5 * (jet_fakes_tau1 + jet_fakes_tau2)
    #var_jet_fakes  = var_jet_fakes_tau1 + var_jet_fakes_tau2

    return jet_fakes, var_jet_fakes