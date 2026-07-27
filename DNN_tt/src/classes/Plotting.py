import math
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
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

    return jet_fakes, var_jet_fakes

def _reorder_for_rowwise_legend(handles, labels, ncol, reverse=False):
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
        cat_title = r'$\tau$ DM: inclusive'
    elif grouping == 'njets':
        ff_dnn_tau1 = 'ff_dnn_tau1_njets'
        ff_dnn_tau2 = 'ff_dnn_tau2_njets'
        cat_title = r'$N_{jets}$: inclusive'
    else:
        ff_dnn_tau1 = 'ff_dnn_tau1'
        ff_dnn_tau2 = 'ff_dnn_tau2'
        cat_title = 'inclusive'

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

    jet_fakes_dnn, var_jet_fakes_dnn = estimate_jet_fakes(
        df,
        bins,
        var,
        ff_dnn_tau1,
        ff_dnn_tau2,
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
        (histograms['wjets']['counts'], '#e76300', r"W+jets"),
        (histograms['embedding']['counts'], '#ffa90e', r'$\tau$ embedded'),
        (histograms['jet_fakes_dnn']['counts'], "#a96b59", r'Jet $\rightarrow \tau_h$'),
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
    handles, labels = _reorder_for_rowwise_legend(handles, labels, ncol=4)
    ax[0].legend(handles, labels, title=' ', title_fontsize=20, loc='upper left', ncol=4, frameon=False)
    adjust_ylim_for_legend(ax[0])
    ax[0].tick_params(direction='in', top=True, right=True)

    CMS_LABEL(ax)
    CMS_CATEGORY_TITLE(ax, title=cat_title)
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
    ax[1].set_xlabel(label)

    return fig, ax, histograms


def plot_closure_incl(
    df,
    incl,
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
        cat_title = r'$\tau$ DM: inclusive'
    elif grouping == 'njets':
        ff_dnn_tau1 = 'ff_dnn_tau1_njets'
        ff_dnn_tau2 = 'ff_dnn_tau2_njets'
        cat_title = r'$N_{jets}$: inclusive'
    else:
        ff_dnn = f'ff_dnn_incl_{incl}'
        cat_title = 'inclusive'

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
        (histograms['wjets']['counts'], '#e76300', r"W+jets"),
        (histograms['embedding']['counts'], '#ffa90e', r'$\tau$ embedded'),
        (histograms['jet_fakes_dnn']['counts'], "#a96b59", r'Jet $\rightarrow \tau_h$'),
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
    handles, labels = _reorder_for_rowwise_legend(handles, labels, ncol=4)
    ax[0].legend(handles, labels, title=' ', title_fontsize=20, loc='upper left', ncol=4, frameon=False)
    adjust_ylim_for_legend(ax[0])
    ax[0].tick_params(direction='in', top=True, right=True)

    CMS_LABEL(ax)
    CMS_CATEGORY_TITLE(ax, title=cat_title)
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

def plot_fake_factors(
        df,
        category_title = None,
        clipped = True
) -> None:
    hep.style.use(hep.style.CMS)
	
    if clipped:
        ff_dnn_tau1 = 'ff_dnn_tau1'
        ff_dnn_tau2 = 'ff_dnn_tau2'
    
        bins_tau1 = np.linspace(0, 1, 51)
        bins_tau2 = np.linspace(0, 1., 51)
    else:
        ff_dnn_tau1 = 'ff_unclipped_dnn_tau1'
        ff_dnn_tau2 = 'ff_unclipped_dnn_tau2'
    
        bins_tau1 = np.linspace(0, 2., 51)
        bins_tau2 = np.linspace(0, 2., 51)
    

    fig, ax = plt.subplots(2, 1, figsize=(11.7, 9.1))

    ax[0].hist(df.data.AR_tau1[ff_dnn_tau1], bins=bins_tau1, histtype = 'step', linewidth = 2, label='Tau 1')
    ax[0].set_ylabel("Events")
    ax[0].legend()

    CMS_CHANNEL_TITLE([ax[0]])
    CMS_LUMI_TITLE([ax[0]])
    CMS_LABEL([ax[0]])
    CMS_CATEGORY_TITLE([ax[0]], title = category_title)

    ax[1].set_ylabel('Events')
    ax[1].hist(df.data.AR_tau2[ff_dnn_tau2], bins=bins_tau2, histtype = 'step', linewidth = 2, label="Tau 2")
    ax[1].set_xlabel("fake_factor")
    ax[1].legend()
    return fig, ax


def plot_fake_factors_combTaus(
        df,
        clipped = True
) -> None:
    hep.style.use(hep.style.CMS)
	
    if clipped:
        ff_dnn_tau1 = 'ff_dnn_tau1'
        ff_dnn_tau2 = 'ff_dnn_tau2'
    
        bins_tau1 = np.linspace(0, 1, 51)
        bins_tau2 = np.linspace(0, 1., 51)
    else:
        ff_dnn_tau1 = 'ff_unclipped_dnn_tau1'
        ff_dnn_tau2 = 'ff_unclipped_dnn_tau2'
    
        bins_tau1 = np.linspace(0, 2., 51)
        bins_tau2 = np.linspace(0, 2., 51)
    

    fig, ax = plt.subplots(2, 1, figsize=(11.7, 9.1))


    n1, binedges = np.histogram(df.data.AR_tau1[ff_dnn_tau1], bins=bins_tau1)
    n2, _ = np.histogram(df.data.AR_tau2[ff_dnn_tau2], bins=bins_tau2)

    n = n1 + n2
        
    fig, ax = plt.subplots(1, 1, figsize=(11.7, 9.1))

    CMS_CHANNEL_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_LABEL(ax)
    #CMS_CATEGORY_TITLE(ax, title=category_title)

    ax.stairs(n, binedges, linewidth=2, label=r'Combined $\tau_h$: incl')

    ax.set_ylabel('Events')
    ax.set_xlabel("fake_factor")
    ax.legend()
    return fig, ax

def plot_fake_factors_incl(
        df,
        incl,
        category_title = None,
        clipped = True
) -> None:
    hep.style.use(hep.style.CMS)
	
    if clipped:
        ff_dnn = f'ff_dnn_incl_{incl}'    
        bins = np.linspace(0, 1, 51)
    else:
        ff_dnn = f'ff_unclipped_dnn_incl_{incl}'    
        bins = np.linspace(0, 2., 51)
    

    fig, ax = plt.subplots(1, 1, figsize=(11.7, 9.1))
    
    CMS_CHANNEL_TITLE([ax])
    CMS_LUMI_TITLE([ax])
    CMS_LABEL([ax])
    CMS_CATEGORY_TITLE([ax], title = category_title)

    n = ax.hist(df.data.AR[ff_dnn], bins=bins, histtype = 'step', linewidth = 2, label='Tau incl.')
    ax.set_ylabel('Events')
    ax.set_xlabel("fake_factor")
    ax.set_ylim(top=1.2*np.max(n[0]))
    ax.legend()

    return fig, ax

def plot_classic_fake_factors(
        df,
        short,
        category_title = None,
) -> None:
    hep.style.use(hep.style.CMS)
	
    ff_tau1 = 'ff_classic_tau1'
    ff_tau2 = 'ff_classic_tau2'
    

    fig, ax = plt.subplots(2, 1, figsize=(11.7, 9.1))

    if short=='jv':
        bins_tau1 = np.linspace(0, 0.5, 71)
        bins_tau2 = np.linspace(0, 0.5, 71)

        n = ax[0].hist(df.data.AR_tau1_jvoss[ff_tau1], bins=bins_tau1, histtype = 'step', linewidth = 2, label='Tau 1')
        ax[1].hist(df.data.AR_tau2_jvoss[ff_tau2], bins=bins_tau2, histtype = 'step', linewidth = 2, label="Tau 2")

    elif short=='sg':
        bins_tau1 = np.linspace(0, 0.5, 51)
        bins_tau2 = np.linspace(0, 0.5, 51)
        n = ax[0].hist(df.data.AR_tau1_sgiappic[ff_tau1], bins=bins_tau1, histtype = 'step', linewidth = 2, label='Tau 1')
        ax[1].hist(df.data.AR_tau2_sgiappic[ff_tau2], bins=bins_tau2, histtype = 'step', linewidth = 2, label="Tau 2")

    else:
        print(f'short = {short} is not implmented. Use either jv or sg')

    ax[0].set_ylabel("Events")
    ax[0].legend()
    ax[0].set_ylim(0, 1.2*np.max(n[0]))

    CMS_CHANNEL_TITLE([ax[0]])
    CMS_LUMI_TITLE([ax[0]])
    CMS_LABEL([ax[0]])
    CMS_CATEGORY_TITLE([ax[0]], title = category_title)

    ax[1].set_ylabel('Events')
    ax[1].set_xlabel("fake_factor")
    ax[1].legend()
    return fig, ax


def plot_fake_factors_in_DR(
        df,
        category_title,
) -> None:

	bins_wjets = np.linspace(0, 1, 51)
	bins_qcd = np.linspace(0, 0.5, 51)

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

def FF_closure_in_DR_tau1(
    df,
	var,
	bins,
	label,
	grouping = 'njets',
):
    hep.style.use(hep.style.CMS)

    if grouping == 'tau_decaymode':
        ff_dnn_tau1 = 'ff_DR_dnn_tau1_tau_dm'
    elif grouping == 'njets':
        ff_dnn_tau1 = 'ff_DR_dnn_tau1_njets'

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

def plot_fake_factors_grouped(df, category_title, grouping='tau_decaymode', clipped = True):
    hep.style.use(hep.style.CMS)

    if clipped:
        bins_tau1 = np.linspace(0, 1.0, 51)
        bins_tau2 = np.linspace(0, 1.0, 51)

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
        bins_tau1 = np.linspace(0, 2., 51)
        bins_tau2 = np.linspace(0, 2., 51)

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
        ff1 = 'ff_dnn_tau1_tau_dm'
        ff2 = 'ff_dnn_tau2_tau_dm'
        grouping = ['tau_decaymode_1', 'tau_decaymode_2']
    elif grouping == 'njets':
        ff1 = 'ff_dnn_tau1_njets'
        ff2 = 'ff_dnn_tau2_njets'
    else:
        raise ValueError(f'Unsupported grouping: {grouping}')

    bins_tau1 = np.linspace(0, 1.0, 51)
    bins_tau2 = np.linspace(0, 1.0, 51)

    frame_tau1 = df.data.AR_tau1
    frame_tau2 = df.data.AR_tau2

    if isinstance(grouping, list):
        group_mask_tau1 = _grouping_masks(frame_tau1, grouping[0])
        group_mask_tau2 = _grouping_masks(frame_tau2, grouping[1])
    else:
        group_mask_tau1 = _grouping_masks(frame_tau1, grouping)
        group_mask_tau2 = _grouping_masks(frame_tau2, grouping)

    n1, binedges = np.histogram(frame_tau1[ff1], bins=bins_tau1)
    n2, _ = np.histogram(frame_tau2[ff2], bins=bins_tau2)

    n = n1 + n2

    n1_split = []
    n2_split = []

    for mask, mask_label in group_mask_tau1:
        h,_ = np.histogram(frame_tau1[ff1][mask], bins=bins_tau1)
        n1_split.append(h)

    for mask, mask_label in group_mask_tau2:
        h,_ = np.histogram(frame_tau2[ff2][mask], bins=bins_tau2)
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

    bins_tau1 = np.linspace(0, 1.0, 51)
    bins_tau2 = np.linspace(0, 1.0, 51)

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

def plot_NN_output_FF(
        NN_output_SR_like,
        NN_output_AR_like,
        FF,
        process = 'Wjets',
        ):

    bins = np.linspace(0, 1, 51)

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


# combinations
def plot_fake_factors_ungrouped_splitAndincl(
        df_split,
        df_incl,
        incl = 'or',
        norm = False
) -> None:
    hep.style.use(hep.style.CMS)
	

    ff_dnn_tau1 = 'ff_dnn_tau1'
    ff_dnn_tau2 = 'ff_dnn_tau2'
    ff_dnn_incl = f'ff_dnn_incl_{incl}'

    bins_tau1 = np.linspace(0, 1., 51)
    bins_tau2 = np.linspace(0, 1., 51)
    bins = np.linspace(0, 1., 51)
    print(bins)
    

    fig, ax = plt.subplots(2, 1, figsize=(11.7, 9.1))


    

    lenff_split = len(df_split.data.AR_tau1[ff_dnn_tau1]) + len(df_split.data.AR_tau2[ff_dnn_tau2])
    lenar_split = len(df_split.data.AR_tau1) + len(df_split.data.AR_tau2)
    lenw_split = np.sum(df_split.data.AR_tau1['weight']) + np.sum(df_split.data.AR_tau2['weight'])
    lenff_incl = len(df_incl.data.AR[ff_dnn_incl])
    lenar_incl = len(df_incl.data.AR)
    lenw_incl = np.sum(df_incl.data.AR['weight'])

    """
    print("tau split:")
    print(lenff_split)
    print(lenar_split)
    print(lenw_split)

    print("tau incl:")
    print(lenff_incl)
    print(lenar_incl)
    print(lenw_incl)
    """

    fig, ax = plt.subplots(1, 1, figsize=(11.7, 9.1))

    CMS_CHANNEL_TITLE(ax)
    CMS_LUMI_TITLE(ax)
    CMS_LABEL(ax)
    CMS_CATEGORY_TITLE(ax, title='inclusive')

    if norm:
        n1, binedges = np.histogram(df_split.data.AR_tau1[ff_dnn_tau1]*0.5, bins=bins_tau1, density=True)
        n2, _ = np.histogram(df_split.data.AR_tau2[ff_dnn_tau2]*0.5, bins=bins_tau2, density=True)
        n = n1 + n2
        ax.stairs(n/2, binedges, linewidth=2, label=r'$\tau_h$ split')
        m = ax.hist(df_incl.data.AR[ff_dnn_incl], bins=bins, density=True, histtype = 'step', linewidth = 2, label=r'$\tau_h$ inclusive')
        print(np.sum(n/2) *0.02)
        print(np.sum(m[0]))
    else:
        n1, binedges = np.histogram(df_split.data.AR_tau1[ff_dnn_tau1]*0.5, bins=bins_tau1)
        n2, _ = np.histogram(df_split.data.AR_tau2[ff_dnn_tau2]*0.5, bins=bins_tau2)
        n = n1 + n2
        ax.stairs(n, binedges, linewidth=2, label=r'$\tau_h$ split')
        m = ax.hist(df_incl.data.AR[ff_dnn_incl], bins=bins, histtype = 'step', linewidth = 2, label=r'$\tau_h$ inclusive')

    ax.set_ylabel('Events')
    ax.set_xlabel("fake_factor")
    ax.set_ylim(top=1.2*np.max([np.max(n), np.max(m[0])]))
    ax.legend()
    return fig, ax

