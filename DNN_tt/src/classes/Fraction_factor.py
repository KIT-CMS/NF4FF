import logging

import numpy as np

from classes.Loading import write_yaml_to_file, load_config

logger = logging.getLogger(__name__)
#pt2_bin_edges = np.array([40, 41, 42 , 43, 44, 45, 46, 48, 50, 55, 60, 65, 70, np.inf])
#pt2_bin_edges = np.array([40, 45, 50 , 55, 60, 65, 70, 75, 80, 90, 100, 120, 200, np.inf])

def fraction_in_bins(df_tau1, df_tau2, frac_file, region='AR_like', pt1_bin_edges=None, pt2_bin_edges=None):
    '''
    df_taun = df.data.AR_like_taun
    '''
    # ----- weights -----
    if region == 'AR_like':
        weights_tau1 = df_tau1["weight_qcd"] * df_tau1["ff_DR_dnn_tau1"]
        weights_tau2 = df_tau2["weight_qcd"] * df_tau2["ff_DR_dnn_tau2"]
    elif region == 'AR':
        weights_tau1 = df_tau1["weight"] * df_tau1["ff_dnn_tau1"]
        weights_tau2 = df_tau2["weight"] * df_tau2["ff_dnn_tau2"]
    else:
        raise ValueError(f"Unknown region: {region!r}. Expected 'AR_like' or 'AR'.")

    # ----- bins -----
    pt2_values = np.concatenate([df_tau1["pt_2"].to_numpy(), df_tau2["pt_2"].to_numpy()])
    weights = np.concatenate([weights_tau1.to_numpy(), weights_tau2.to_numpy()])

    if pt2_bin_edges is not None or pt1_bin_edges is not None:
        pt1_bin_edges = pt1_bin_edges
        pt2_bin_edges = pt2_bin_edges
    else:
        pt2_bin_edges = _equal_weight_bin_edges(
            pt2_values,
            weights,
            events_per_bin=5000,
        )
        pt1_bin_edges = pt2_bin_edges

    # ----- counts -----
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

    fraction = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator != 0,
    )

    # ----- global -----
    h = fraction.flatten()
    h = h[~np.isnan(h)]
    global_frac = np.mean(h)
    std = np.std(h)

    # ----- save fraction in yaml for plotting -----
    all_frac = load_config(frac_file)
    all_frac[region]['ungrouped'] = dict(zip(['fraction', 'pt1_edges', 'pt2_edges', 'global_frac', 'global_std'],[fraction, pt1_edges, pt2_edges, global_frac, std]))
    write_yaml_to_file(all_frac, frac_file)

    return fraction, pt1_edges, pt2_edges

def fraction_in_bins_grouped(
        df_tau1, 
        df_tau2,
        frac_file: str,
        region='AR_like', 
        pt1_bin_edges=None, pt2_bin_edges=None,
        grouping=None, grouping_variable=None, grouping_definition=None,):
    '''
    Calculate the tau-1 fraction independently for every requested group.

    ``grouping_variable`` may either be one column name (for example
    ``"njets"``) or the two column names belonging to tau 1 and tau 2 (for
    example ``["tau_decaymode_1", "tau_decaymode_2"]``).  The return value is
    a dictionary mapping the group name to the usual
    ``(fraction, pt1_edges, pt2_edges)`` tuple.
    '''

    if grouping_variable is None or grouping_definition is None:
        logger.warning("Grouping variable or grouping definition is None. Calculating ungrouped fractions instead.")
        return fraction_in_bins(df_tau1, df_tau2, region, pt1_bin_edges, pt2_bin_edges)

    # ----- grouping variable handling -----
    if isinstance(grouping_variable, list):
        if len(grouping_variable) != 2:
            raise ValueError("grouping_variable must contain exactly two column names when supplied as a list.")
        grouping_var_1 = grouping_variable[0]
        grouping_var_2 = grouping_variable[1]
    else:
        grouping_var_1 = grouping_variable
        grouping_var_2 = grouping_variable    

    if region not in {'AR_like', 'AR'}:
        raise ValueError(f"Unknown region: {region!r}. Expected 'AR_like' or 'AR'.")

    group_tau1_masks = _build_group_masks(
        np.asarray(df_tau1[grouping_var_1]), grouping_definition
    )
    group_tau2_masks = dict(_build_group_masks(
        np.asarray(df_tau2[grouping_var_2]), grouping_definition
    ))

    grouped_fractions = {}
    for group_name, tau1_mask in group_tau1_masks:
        tau2_mask = group_tau2_masks[group_name]
        if not np.any(tau1_mask) and not np.any(tau2_mask):
            raise ValueError(f"Group {group_name!r} contains no events.")
        
        # ----- fraction in bins for grouped -----
        df1 = df_tau1.loc[tau1_mask]
        df2 = df_tau2.loc[tau2_mask]

        # ----- weights -----
        if region == 'AR_like':
            weights_tau1 = df1["weight_qcd"] * df1[f"ff_DR_dnn_tau1_{grouping}"]
            weights_tau2 = df2["weight_qcd"] * df2[f"ff_DR_dnn_tau2_{grouping}"]
        elif region == 'AR':
            weights_tau1 = df1["weight"] * df1[f"ff_dnn_tau1_{grouping}"]
            weights_tau2 = df2["weight"] * df2[f"ff_dnn_tau2_{grouping}"]
        else:
            raise ValueError(f"Unknown region: {region!r}. Expected 'AR_like' or 'AR'.")
    
        # ----- bins -----
        pt2_values = np.concatenate([df1["pt_2"].to_numpy(), df2["pt_2"].to_numpy()])
        weights = np.concatenate([weights_tau1.to_numpy(), weights_tau2.to_numpy()])
    
        if pt2_bin_edges is not None or pt1_bin_edges is not None:
            pt1_bin_edges = pt1_bin_edges
            pt2_bin_edges = pt2_bin_edges
        else:
            pt2_bin_edges = _equal_weight_bin_edges(
                pt2_values,
                weights,
                events_per_bin=5000,
            )
            pt1_bin_edges = pt2_bin_edges
    
        # ----- counts -----
        f1_t2, pt1_edges, pt2_edges = np.histogram2d(
            df1["pt_1"],
            df1["pt_2"],
            bins=(pt1_bin_edges, pt2_bin_edges),
            weights=weights_tau1,
        )
    
        t1_f2, _, _ = np.histogram2d(
            df2["pt_1"],
            df2["pt_2"],
            bins=(pt1_bin_edges, pt2_bin_edges),
            weights=weights_tau2,
        )
    
        numerator = f1_t2
        denominator = f1_t2 + t1_f2
    
        fraction = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator != 0,
        )

        grouped_fractions[group_name] = fraction, pt1_edges, pt2_edges
    
        # ----- global -----
        h = fraction.flatten()
        h = h[~np.isnan(h)]
        global_frac = np.mean(h)
        std = np.std(h)

    #print(grouped_fractions)
    # ----- save fraction in yaml for plotting -----
    all_frac = load_config(frac_file)
    all_frac[region][f'{grouping}'] = dict(zip(['fraction', 'pt1_edges', 'pt2_edges', 'global_frac', 'global_std'],[grouped_fractions, pt1_edges, pt2_edges, global_frac, std]))
    write_yaml_to_file(all_frac, frac_file)

    logger.info("Calculated fraction factors for group %s", group_name)

    return grouped_fractions


def _build_group_masks(values, grouping_definition):
    """Build masks using the same group semantics as grouped fake factors."""
    masks = []
    for group in grouping_definition:
        if len(group) == 1:
            value = group[0]
            group_name = f"{value}"
            mask = values == value
        elif len(group) == 2:
            low, high = group
            group_name = f"{low}_{high}"
            mask = (values >= low) & (values <= high)
        else:
            raise ValueError(f"Invalid group definition: {group}")
        masks.append((group_name, mask))
    return masks

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


def fraction_for_events_grouped(
        frame,
        grouped_fractions,
        grouping_variable,
        grouping_definition,
):
    """Look up the appropriate grouped fraction for every event."""
    if not isinstance(grouped_fractions, dict):
        raise TypeError(
            "grouped_fractions must be the dictionary returned by "
            "fraction_in_bins_grouped()."
        )

    event_fractions = np.full(len(frame), 0.5, dtype=float)
    assigned = np.zeros(len(frame), dtype=bool)
    group_masks = _build_group_masks(
        np.asarray(frame[grouping_variable]), grouping_definition
    )

    for group_name, group_mask in group_masks:
        if group_name not in grouped_fractions:
            raise KeyError(f"No fraction histogram for group {group_name!r}.")
        if not np.any(group_mask):
            continue

        group_frac, pt1_edges, pt2_edges = grouped_fractions[group_name]
        event_fractions[group_mask] = fractions_for_events(
            frame.loc[group_mask], group_frac, pt1_edges, pt2_edges
        )
        assigned |= group_mask

    if np.any(~assigned):
        logger.warning(
            "%d events are outside the fraction grouping definition; "
            "using the fallback fraction 0.5.",
            np.count_nonzero(~assigned),
        )

    return event_fractions

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

def _equal_count_bin_edges(values, events_per_bin=1000):
    values = np.asarray(values)
    values = np.sort(values[np.isfinite(values)])

    if values.size == 0:
        raise ValueError("Cannot calculate bin edges from an empty sample.")

    edges = values[::events_per_bin]
    edges = np.unique(np.concatenate(([values[0]], edges, [np.inf])))

    return edges

def _equal_weight_bin_edges(values, weights, events_per_bin=1000):
    """Return edges whose bins contain approximately equal absolute weight.

    ``events_per_bin`` determines the target number of bins, as in
    :func:`equal_count_bin_edges`.  Absolute weights are used because QCD
    subtraction weights can be signed and a signed cumulative distribution is
    not suitable for defining quantiles.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)

    if values.shape != weights.shape:
        raise ValueError("Values and weights must have the same shape.")
    if events_per_bin <= 0:
        raise ValueError("events_per_bin must be positive.")

    finite = np.isfinite(values) & np.isfinite(weights)
    values = values[finite]
    weights = np.abs(weights[finite])

    if values.size == 0:
        raise ValueError("Cannot calculate bin edges from an empty sample.")
    if not np.any(weights > 0):
        raise ValueError("Cannot calculate weighted bin edges from zero weights.")

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    n_bins = max(1, int(np.ceil(values.size / events_per_bin)))
    cumulative_weight = np.cumsum(weights)
    targets = cumulative_weight[-1] * np.arange(1, n_bins) / n_bins
    internal_edges = values[np.searchsorted(cumulative_weight, targets, side="left")]

    return np.unique(np.concatenate(([values[0]], internal_edges, [np.inf])))

