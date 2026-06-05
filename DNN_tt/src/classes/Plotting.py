import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
):
    if grouping == 'tau_decaymode':
        ff_dnn = 'ff_dnn_tdm'
    elif grouping == 'njets':
        ff_dnn = 'ff_dnn_njets'

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
    CMS_CATEGORY_TITLE(ax)
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


    return fig, ax, histograms


def plot_fake_factors(
        df,
        category_title,
		grouping = 'tau_decaymode',
) -> None:

	if grouping == 'tau_decaymode':
		ff_dnn_qcd = 'ff_dnn_qcd_tdm'
		ff_dnn_wjets = 'ff_dnn_wjets_tdm'
	elif grouping == 'njets':
		ff_dnn_qcd = 'ff_dnn_qcd_njets'
		ff_dnn_wjets = 'ff_dnn_wjets_njets'
    

	bins_wjets = np.linspace(0, 1, 50)
	bins_qcd = np.linspace(0, 0.5, 50)

	fig, ax = plt.subplots(2, 1, figsize=(10, 7))
	ax[0].hist(df.data.AR[ff_dnn_wjets], bins=bins_wjets, histtype = 'step', linewidth = 2, label='Wjets: t_dm incl')
	ax[0].hist(df.data.AR[ff_dnn_wjets][df.data.AR.tau_decaymode_2 == 1], bins=bins_wjets, histtype = 'step', ls = '--', label='Wjets: t_dm = 0')
	ax[0].hist(df.data.AR[ff_dnn_wjets][df.data.AR.tau_decaymode_2 == 0], bins=bins_wjets, histtype = 'step', ls = '--', label='Wjets: t_dm = 1')
	ax[0].hist(df.data.AR[ff_dnn_wjets][df.data.AR.tau_decaymode_2 == 10], bins=bins_wjets, histtype = 'step', ls = '--', label='Wjets: t_dm = 10')
	ax[0].hist(df.data.AR[ff_dnn_wjets][df.data.AR.tau_decaymode_2 == 11], bins=bins_wjets, histtype = 'step', ls = '--', label='Wjets: t_dm = 11')
	ax[0].set_ylabel("Events")
	ax[0].legend()
	ax[0].set_ylim(0, 33000)

	CMS_CHANNEL_TITLE([ax[0]])
	CMS_LUMI_TITLE([ax[0]])
	CMS_LABEL([ax[0]])
	CMS_CATEGORY_TITLE([ax[0]], title = category_title)

	ax[1].set_ylabel('Events')
	ax[1].hist(df.data.AR[ff_dnn_qcd], bins=bins_qcd, histtype = 'step', linewidth = 2, label="QCD: t_dm: incl")
	ax[1].hist(df.data.AR[ff_dnn_qcd][df.data.AR.tau_decaymode_2 == 0], bins=bins_qcd, histtype = 'step', ls = '--', label = "QCD: t_dm = 0")
	ax[1].hist(df.data.AR[ff_dnn_qcd][df.data.AR.tau_decaymode_2 == 1], bins=bins_qcd, histtype = 'step', ls = '--', label = "QCD: t_dm = 1")
	ax[1].hist(df.data.AR[ff_dnn_qcd][df.data.AR.tau_decaymode_2 == 10], bins=bins_qcd, histtype = 'step', ls = '--', label = "QCD : t_dm = 10")
	ax[1].hist(df.data.AR[ff_dnn_qcd][df.data.AR.tau_decaymode_2 == 11], bins=bins_qcd, histtype = 'step', ls = '--', label = "QCD : t_dm = 11")

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
	ax[0].hist(df.data.AR_like_wjets.ff_dnn_wjets_tdm, bins=bins_wjets, histtype = 'step', linewidth = 2, label='Wjets: t_dm incl')
	ax[0].hist(df.data.AR_like_wjets.ff_dnn_wjets_tdm[df.data.AR_like_wjets.tau_decaymode_2 == 1], bins=bins_wjets, histtype = 'step', ls = '--', label='Wjets: t_dm = 0')
	ax[0].hist(df.data.AR_like_wjets.ff_dnn_wjets_tdm[df.data.AR_like_wjets.tau_decaymode_2 == 0], bins=bins_wjets, histtype = 'step', ls = '--', label='Wjets: t_dm = 1')
	ax[0].hist(df.data.AR_like_wjets.ff_dnn_wjets_tdm[df.data.AR_like_wjets.tau_decaymode_2 == 10], bins=bins_wjets, histtype = 'step', ls = '--', label='Wjets: t_dm = 10')
	ax[0].hist(df.data.AR_like_wjets.ff_dnn_wjets_tdm[df.data.AR_like_wjets.tau_decaymode_2 == 11], bins=bins_wjets, histtype = 'step', ls = '--', label='Wjets: t_dm = 11')
	ax[0].set_ylabel("Events")
	ax[0].legend()
	ax[0].set_ylim(0, 20000)

	CMS_CHANNEL_TITLE([ax[0]])
	CMS_LUMI_TITLE([ax[0]])
	CMS_LABEL([ax[0]])
	CMS_CATEGORY_TITLE([ax[0]], title = category_title)

	ax[1].set_ylabel('Events')
	ax[1].hist(df.data.AR_like_qcd.ff_dnn_qcd_tdm, bins=bins_qcd, histtype = 'step', linewidth = 2, label="QCD: t_dm: incl")
	ax[1].hist(df.data.AR_like_qcd.ff_dnn_qcd_tdm[df.data.AR_like_qcd.tau_decaymode_2 == 0], bins=bins_qcd, histtype = 'step', ls = '--', label = "QCD: t_dm = 0")
	ax[1].hist(df.data.AR_like_qcd.ff_dnn_qcd_tdm[df.data.AR_like_qcd.tau_decaymode_2 == 1], bins=bins_qcd, histtype = 'step', ls = '--', label = "QCD: t_dm = 1")
	ax[1].hist(df.data.AR_like_qcd.ff_dnn_qcd_tdm[df.data.AR_like_qcd.tau_decaymode_2 == 10], bins=bins_qcd, histtype = 'step', ls = '--', label = "QCD : t_dm = 10")
	ax[1].hist(df.data.AR_like_qcd.ff_dnn_qcd_tdm[df.data.AR_like_qcd.tau_decaymode_2 == 11], bins=bins_qcd, histtype = 'step', ls = '--', label = "QCD : t_dm = 11")

	ax[1].set_xlabel("fake_factor")
	ax[1].legend()
	return fig, ax

def FF_closure_in_DR_wjets(
    df,
	var,
	bins,
	label,
	grouping = 'tau_decaymode',
):
	if grouping == 'tau_decaymode':
		ff_dnn_wjets = 'ff_dnn_wjets_tdm'
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

	ax[0].set_xlabel(var)
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

def FF_closure_in_DR_qcd(
    df,
	var,
	bins,
	label,
	grouping = 'tau_decaymode',
):
	if grouping == 'tau_decaymode':
		ff_dnn_qcd = 'ff_dnn_qcd_tdm'
	elif grouping == 'njets':
		ff_dnn_qcd = 'ff_dnn_qcd_njets'
    
	counts_SR_like, bin_edges = np.histogram(df.data.SR_like_qcd[var], weights = df.data.SR_like_qcd.weight_qcd, bins = bins)
	counts_FF_AR_like, _ = np.histogram(df.data.AR_like_qcd[var], weights = df.data.AR_like_qcd.weight_qcd * df.data.AR_like_qcd[ff_dnn_qcd], bins = bins)

	variance_SR_like, _ = np.histogram(df.data.SR_like_qcd[var], weights = df.data.SR_like_qcd.weight_qcd**2, bins = bins)
	variance_FF_AR_like, _ = np.histogram(
		df.data.AR_like_qcd[var], 
		weights = (df.data.AR_like_qcd.weight_qcd * df.data.AR_like_qcd[ff_dnn_qcd])**2,
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

	ax[0].set_xlabel(var)
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
	ax[1].set_label(label)

	return fig, ax


def _grouping_masks(frame, grouping):
	if grouping == 'tau_decaymode':
		return [
			(frame.tau_decaymode_2 == 0, 't_dm = 0'),
			(frame.tau_decaymode_2 == 1, 't_dm = 1'),
			(frame.tau_decaymode_2 == 10, 't_dm = 10'),
			(frame.tau_decaymode_2 == 11, 't_dm = 11'),
		]
	if grouping == 'njets':
		return [
			(frame.njets == 0, 'njets = 0'),
			(frame.njets == 1, 'njets = 1'),
			(frame.njets >= 2, 'njets >= 2'),
		]
	raise ValueError(f'Unsupported grouping: {grouping}')


def plot_fake_factors_grouped(df, category_title, grouping='tau_decaymode'):
	if grouping == 'tau_decaymode':
		ff_wjets = 'ff_dnn_wjets_tdm'
		ff_qcd = 'ff_dnn_qcd_tdm'
	elif grouping == 'njets':
		ff_wjets = 'ff_dnn_wjets_njets'
		ff_qcd = 'ff_dnn_qcd_njets'
	else:
		raise ValueError(f'Unsupported grouping: {grouping}')

	bins_wjets = np.linspace(0, 1, 50)
	bins_qcd = np.linspace(0, 0.5, 50)
	frame = df.data.AR

	fig, ax = plt.subplots(2, 1, figsize=(10, 7))
	ax[0].hist(frame[ff_wjets], bins=bins_wjets, histtype='step', linewidth=2, label='Wjets: incl')
	for mask, mask_label in _grouping_masks(frame, grouping):
		ax[0].hist(frame[ff_wjets][mask], bins=bins_wjets, histtype='step', ls='--', label=f'Wjets: {mask_label}')
	ax[0].set_ylabel('Events')
	ax[0].legend()
	ax[0].set_ylim(0, 33000)

	CMS_CHANNEL_TITLE([ax[0]])
	CMS_LUMI_TITLE([ax[0]])
	CMS_LABEL([ax[0]])
	CMS_CATEGORY_TITLE([ax[0]], title=category_title)

	ax[1].set_ylabel('Events')
	ax[1].hist(frame[ff_qcd], bins=bins_qcd, histtype='step', linewidth=2, label='QCD: incl')
	for mask, mask_label in _grouping_masks(frame, grouping):
		ax[1].hist(frame[ff_qcd][mask], bins=bins_qcd, histtype='step', ls='--', label=f'QCD: {mask_label}')
	ax[1].set_xlabel('fake_factor')
	ax[1].legend()
	return fig, ax


def plot_fake_factors_in_dr_grouped(df, category_title, grouping='tau_decaymode'):
	if grouping == 'tau_decaymode':
		ff_wjets = 'ff_dnn_wjets_tdm'
		ff_qcd = 'ff_dnn_qcd_tdm'
	elif grouping == 'njets':
		ff_wjets = 'ff_dnn_wjets_njets'
		ff_qcd = 'ff_dnn_qcd_njets'
	else:
		raise ValueError(f'Unsupported grouping: {grouping}')

	bins_wjets = np.linspace(0, 1, 50)
	bins_qcd = np.linspace(0, 0.5, 50)
	frame_wjets = df.data.AR_like_wjets
	frame_qcd = df.data.AR_like_qcd

	fig, ax = plt.subplots(2, 1, figsize=(10, 7))
	ax[0].hist(frame_wjets[ff_wjets], bins=bins_wjets, histtype='step', linewidth=2, label='Wjets: incl')
	for mask, mask_label in _grouping_masks(frame_wjets, grouping):
		ax[0].hist(frame_wjets[ff_wjets][mask], bins=bins_wjets, histtype='step', ls='--', label=f'Wjets: {mask_label}')
	ax[0].set_ylabel('Events')
	ax[0].legend()
	ax[0].set_ylim(0, 20000)

	CMS_CHANNEL_TITLE([ax[0]])
	CMS_LUMI_TITLE([ax[0]])
	CMS_LABEL([ax[0]])
	CMS_CATEGORY_TITLE([ax[0]], title=category_title)

	ax[1].set_ylabel('Events')
	ax[1].hist(frame_qcd[ff_qcd], bins=bins_qcd, histtype='step', linewidth=2, label='QCD: incl')
	for mask, mask_label in _grouping_masks(frame_qcd, grouping):
		ax[1].hist(frame_qcd[ff_qcd][mask], bins=bins_qcd, histtype='step', ls='--', label=f'QCD: {mask_label}')
	ax[1].set_xlabel('fake_factor')
	ax[1].legend()
	return fig, ax
