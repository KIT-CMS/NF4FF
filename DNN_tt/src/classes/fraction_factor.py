import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def pt_mask(df):
    bin_edges = [40, 45, 50 , 55, 60, 65, 70, 75, 80, 90, 100, 120, 200]
    n_bins = len(bin_edges) -1

    masks = np.empty((n_bins, n_bins))
    for i in range(n_bins):
        if i == np.max(range(n_bins)):
            mask_pt2 = ((bin_edges[i] <= df["pt_2"]))
        else:
            mask_pt2 = ((bin_edges[i] <= df["pt_2"]) & (df["pt_2"] < bin_edges[i + 1]))        

        for j in range(n_bins):
            if j == np.max(range(n_bins)):
                mask_pt1 = ((bin_edges[j] <= df["pt_1"]))
            else:
                mask_pt1 = ((bin_edges[j] <= df["pt_1"]) & (df["pt_1"] < bin_edges[j + 1]))

            masks[i, j] = mask_pt1 & mask_pt2

    return masks


def fraction_in_bins(df):
    bin_edges = np.array([40, 45, 50 , 55, 60, 65, 70, 75, 80, 90, 100, 120, 200, np.inf])

    f1_t2_mask = ((df["id_tau_vsJet_Tight_1"] < 0.5) & (df["id_tau_vsJet_Tight_2"] > 0.5))

    t1_f2_mask = ((df["id_tau_vsJet_Tight_1"] > 0.5)  & (df["id_tau_vsJet_Tight_2"] < 0.5))

    numerator_mask = f1_t2_mask
    denominator_mask = (f1_t2_mask | t1_f2_mask)

    numerator, pt1_edges, pt2_edges = np.histogram2d(
        df.loc[numerator_mask, "pt_1"],
        df.loc[numerator_mask, "pt_2"],
        bins=(bin_edges, bin_edges),
    )

    t1_f2, _, _ = np.histogram2d(
        df.loc[t1_f2_mask, "pt_1"],
        df.loc[t1_f2_mask, "pt_2"],
        bins=(bin_edges, bin_edges),
    )

    denominator, _, _ = np.histogram2d(
        df.loc[denominator_mask, "pt_1"],
        df.loc[denominator_mask, "pt_2"],
        bins=(bin_edges, bin_edges),
    )

    #print(t1_f2)
    #print("Numerator:\n", numerator)
    #print("Denominator:\n", denominator)


    fraction = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator != 0,
    )

    return fraction, pt1_edges, pt2_edges



def plot_fractions(df):
    frac, pt1_edges, pt2_edges = fraction_in_bins(df)

    n_pt1, n_pt2 = frac.shape

    fig, ax = plt.subplots(1, 1, figsize=(11.7, 9.1))
    image = ax.imshow(
        frac.T,
        origin="lower",
        aspect="equal",
        interpolation="none",
        cmap="viridis",
        extent=(-0.5, n_pt1 - 0.5, -0.5, n_pt2 - 0.5),
        vmin=np.nanmin(frac.T),
        vmax=np.nanmax(frac.T),
    )

    for pt2_bin in range(n_pt2):
        for pt1_bin in range(n_pt1):
            value = frac.T[pt2_bin, pt1_bin]

            if np.isfinite(value):
                ax.text(
                    pt1_bin,
                    pt2_bin,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black" if value > 0.42 else "white",
                    fontsize=8,
                )

    # Positions of bin boundaries.
    x_boundaries = np.arange(n_pt1 + 1) - 0.5
    y_boundaries = np.arange(n_pt2 + 1) - 0.5

    # Format the actual pT bin edges.
    edge_labels = [
        f"{edge:g}" if np.isfinite(edge) else "∞"
        for edge in pt1_edges
    ]

    ax.set_xticks(x_boundaries)
    ax.set_yticks(y_boundaries)
    ax.set_xticklabels(edge_labels)#, rotation=45, ha="right")
    ax.set_yticklabels(edge_labels)

    # Draw lines along the square boundaries.
    ax.grid(
        which="major",
        color="white",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.set_xlabel(r"$p_{T,1}$ bin [GeV]")
    ax.set_ylabel(r"$p_{T,2}$ bin [GeV]")

    fig.colorbar(image, ax=ax, label="Fraction factor")
    fig.tight_layout()
    plt.savefig('/work/tapp/TauFF/NF4FF/DNN_tt/src/classes/fraction_test.png')
    plt.close(fig)