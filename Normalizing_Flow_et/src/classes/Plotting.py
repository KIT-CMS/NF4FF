import math
import matplotlib.pyplot as plt
import numpy as np

def CMS_CHANNEL_TITLE(ax, *args, **kwargs):
    ax[0].set_title(
        r"$e\tau_h$",
        fontsize=20,
        loc="left",
        fontproperties="Tex Gyre Heros"
    )

def CMS_NJETS_TITLE(ax, title=r"$N_{jets} \geq 0$", *args, **kwargs):
    ax[0].set_title(
        title,
        fontsize=20,
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
