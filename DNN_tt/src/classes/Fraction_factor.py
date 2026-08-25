import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def equal_count_bin_edges(values, events_per_bin=1000):
    values = np.asarray(values)
    values = np.sort(values[np.isfinite(values)])

    if values.size == 0:
        raise ValueError("Cannot calculate bin edges from an empty sample.")

    edges = values[::events_per_bin]
    edges = np.unique(np.concatenate(([values[0]], edges, [np.inf])))

    return edges

def fraction_in_bins(df_tau1, df_tau2, plotting=False, region='DR'):
    '''
    df_taun = df.data.AR_like_taun
    '''

    pt1_values = np.concatenate([df_tau1["pt_1"].to_numpy(), df_tau2["pt_1"].to_numpy()])

    if plotting:
        pt1_bin_edges = np.array([40, 45, 50 , 55, 60, 65, 70, 75, 80, 90, 100, 120, 200, np.inf])
    else:
        pt1_bin_edges = equal_count_bin_edges(pt1_values, events_per_bin=1000)
    pt2_bin_edges = pt1_bin_edges

    if region == 'DR':
        weights_tau1 = df_tau1["weight_qcd"]
        weights_tau2 = df_tau2["weight_qcd"]
    elif region == 'SR':
        weights_tau1 = df_tau1["weight"]
        weights_tau2 = df_tau2["weight"]
    
    f1_t2, pt1_edges, pt2_edges = np.histogram2d(
        df_tau1["pt_1"],
        df_tau1["pt_2"],
        bins=(pt1_bin_edges, pt2_bin_edges),
        weights=weights_tau1,
    )

    t1_f2, _, _ = np.histogram2d(
        df_tau2["pt_1"],
        df_tau2["pt_2"],
        bins=(pt1_bin_edges, pt2_bin_edges),
        weights=weights_tau2,
    )

    numerator = f1_t2
    denominator = f1_t2 + t1_f2

    #print("Numerator:\n", numerator)
    #print("Denominator:\n", denominator)


    fraction = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator != 0,
    )
    #print(fraction)
    h = fraction.flatten()
    h = h[~np.isnan(h)]
    global_frac = np.sum(h)/len(h)
    print('Global fraction:', global_frac)


    return fraction, pt1_edges, pt2_edges

def fractions_for_events(frame, frac, pt1_edges, pt2_edges):
    pt1_bin = np.searchsorted(
        pt1_edges, frame["pt_1"].to_numpy(), side="right"
    ) - 1
    pt2_bin = np.searchsorted(
        pt2_edges, frame["pt_2"].to_numpy(), side="right"
    ) - 1


    # Protect against values outside the histogram range.
    pt1_bin = np.clip(pt1_bin, 0, frac.shape[0] - 1)
    pt2_bin = np.clip(pt2_bin, 0, frac.shape[1] - 1)

    event_fractions = frac[pt1_bin, pt2_bin]

    # Choose a fallback for bins without AR-like events.
    return np.nan_to_num(event_fractions, nan=0.5)

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

def fraction_in_bins_old(df_tau1, df_tau2, plotting=False):
    f1_t2_mask = ((df_tau1["id_tau_vsJet_Tight_1"] < 0.5) & (df_tau1["id_tau_vsJet_Tight_2"] > 0.5))
    t1_f2_mask = ((df_tau2["id_tau_vsJet_Tight_1"] > 0.5)  & (df_tau2["id_tau_vsJet_Tight_2"] < 0.5))
    numerator_mask = f1_t2_mask

    pt1_values = np.concatenate([
        df_tau1.loc[f1_t2_mask, "pt_1"].to_numpy(),
        df_tau2.loc[t1_f2_mask, "pt_1"].to_numpy(),
    ])

    pt2_values = np.concatenate([
        df_tau1.loc[f1_t2_mask, "pt_2"].to_numpy(),
        df_tau2.loc[t1_f2_mask, "pt_2"].to_numpy(),
    ])

    if plotting:
        pt1_bin_edges = np.array([40, 45, 50 , 55, 60, 65, 70, 75, 80, 90, 100, 120, 200, np.inf])
    else:
        pt1_bin_edges = equal_count_bin_edges(pt1_values, events_per_bin=1000)
    pt2_bin_edges = pt1_bin_edges
    #print(pt1_bin_edges)
    #print(len(pt1_bin_edges))
    #print(pt2_bin_edges)
    #print(len(pt2_bin_edges))

    f1_t2, pt1_edges, pt2_edges = np.histogram2d(
        df_tau1.loc[numerator_mask, "pt_1"],
        df_tau1.loc[numerator_mask, "pt_2"],
        bins=(pt1_bin_edges, pt2_bin_edges),
        #weights=(df_tau1.loc[f1_t2_mask, "weight_qcd"], df_tau1.loc[f1_t2_mask, "weight_qcd"]),
    )

    t1_f2, _, _ = np.histogram2d(
        df_tau2.loc[t1_f2_mask, "pt_1"],
        df_tau2.loc[t1_f2_mask, "pt_2"],
        bins=(pt1_bin_edges, pt2_bin_edges),
        #weights=(df_tau2.loc[t1_f2_mask, "weight_qcd"], df_tau2.loc[t1_f2_mask, "weight_qcd"]),
    )

    numerator = f1_t2
    denominator = f1_t2 + t1_f2

    #print("Numerator:\n", numerator)
    #print("Denominator:\n", denominator)


    fraction = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator != 0,
    )
    #print(fraction)
    h = fraction.flatten()
    h = h[~np.isnan(h)]
    global_frac = np.sum(h)/len(h)
    print('Global fraction:', global_frac)


    return fraction, pt1_edges, pt2_edges