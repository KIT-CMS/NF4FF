import logging

import numpy as np

from classes.Loading import write_yaml_to_file, load_config

logger = logging.getLogger(__name__)

def fraction_in_bins(df_tau1, df_tau2, frac_file, var1='pt_1', var2='pt_2', region='AR_like', var1_bin_edges=None, var2_bin_edges=None):
    '''
    df_taun = df.data.AR_like_taun
    '''
    # ----- weights -----
    if region == 'AR_like':
        weights_tau1 = df_tau1["weight_qcd"] * df_tau1["ff_DR_dnn_tau1"]
        weights_tau2 = df_tau2["weight_qcd"] * df_tau2["ff_DR_dnn_tau2"]
    elif region == 'AR':
        weights_tau1 = df_tau1["weight"] * df_tau1["ff_dnn_tau1_raw"]
        weights_tau2 = df_tau2["weight"] * df_tau2["ff_dnn_tau2_raw"]
    else:
        raise ValueError(f"Unknown region: {region!r}. Expected 'AR_like' or 'AR'.")

    # ----- bins -----
    var2_values = np.concatenate([df_tau1[var2].to_numpy(), df_tau2[var2].to_numpy()])
    weights = np.concatenate([weights_tau1.to_numpy(), weights_tau2.to_numpy()])

    if var2_bin_edges is not None or var1_bin_edges is not None:
        var1_bin_edges = var1_bin_edges
        var2_bin_edges = var2_bin_edges
    else:
        var2_bin_edges = _equal_weight_bin_edges(
            var2_values,
            weights,
            events_per_bin=5000,
        )
        var1_bin_edges = var2_bin_edges

    # ----- counts -----
    f1_t2, var1_edges, var2_edges = np.histogram2d(
        df_tau1[var1],
        df_tau1[var2],
        bins=(var1_bin_edges, var2_bin_edges),
        weights=weights_tau1,
    )

    t1_f2, _, _ = np.histogram2d(
        df_tau2[var1],
        df_tau2[var2],
        bins=(var1_bin_edges, var2_bin_edges),
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
    if var1=='pt_1' and var2=='pt_2':
        all_frac[region]['ungrouped'] = dict(zip(['fraction', 'pt1_edges', 'pt2_edges', 'global_frac', 'global_std'],[fraction, var1_edges, var2_edges, global_frac, std]))
        write_yaml_to_file(all_frac, frac_file)

    return fraction, var1_edges, var2_edges, global_frac, std

def fraction_in_bins_3split(which_frac, df_tau1, df_tau2, df_tau3, frac_file, var1='pt_1', var2='pt_2', region='AR_like', var1_bin_edges=None, var2_bin_edges=None):
    '''
    which_frac = tau1, tau2, tau1&2
    df_taun = df.data.AR_like_taun
    '''
    # ----- weights -----
    if region == 'AR_like':
        weights_tau1 = df_tau1["weight_qcd"] * df_tau1["ff_DR_dnn_1"]
        weights_tau2 = df_tau2["weight_qcd"] * df_tau2["ff_DR_dnn_2"]
        weights_tau3 = df_tau3["weight_qcd"] * df_tau3["ff_DR_dnn_3"]
    elif region == 'AR':
        weights_tau1 = df_tau1["weight"] * df_tau1["ff_dnn_1_raw"]
        weights_tau2 = df_tau2["weight"] * df_tau2["ff_dnn_2_raw"]
        weights_tau3 = df_tau3["weight"] * df_tau3["ff_dnn_3_raw"]
    else:
        raise ValueError(f"Unknown region: {region!r}. Expected 'AR_like' or 'AR'.")

    # ----- bins -----
    var2_values = np.concatenate([df_tau1[var2].to_numpy(), df_tau2[var2].to_numpy(), df_tau3[var2].to_numpy()])
    weights = np.concatenate([weights_tau1.to_numpy(), weights_tau2.to_numpy(), weights_tau3.to_numpy()])

    if var2_bin_edges is not None or var1_bin_edges is not None:
        var1_bin_edges = var1_bin_edges
        var2_bin_edges = var2_bin_edges
    else:
        var2_bin_edges = _equal_weight_bin_edges(
            var2_values,
            weights,
            events_per_bin=8000,
        )
        var1_bin_edges = var2_bin_edges

    # ----- counts -----
    f1_t2, var1_edges, var2_edges = np.histogram2d(
        df_tau1[var1],
        df_tau1[var2],
        bins=(var1_bin_edges, var2_bin_edges),
        weights=weights_tau1,
    )

    t1_f2, _, _ = np.histogram2d(
        df_tau2[var1],
        df_tau2[var2],
        bins=(var1_bin_edges, var2_bin_edges),
        weights=weights_tau2,
    )

    f1_f2, _, _ = np.histogram2d(
            df_tau3[var1],
            df_tau3[var2],
            bins=(var1_bin_edges, var2_bin_edges),
            weights=weights_tau3,
        )

    if which_frac=='tau1': numerator = f1_t2
    elif which_frac=='tau2': numerator = t1_f2
    elif which_frac=='tau1&2': numerator = f1_f2
    else: logger.error(f'which_frac is {which_frac}. Expected tau1, tau2 or tau1&2.')
    denominator = f1_t2 + t1_f2 + f1_f2

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
    if var1=='pt_1' and var2=='pt_2':
        all_frac[region]['ungrouped'] = dict(zip(['fraction', 'pt1_edges', 'pt2_edges', 'global_frac', 'global_std'],[fraction, var1_edges, var2_edges, global_frac, std]))
        write_yaml_to_file(all_frac, frac_file)

    return fraction, var1_edges, var2_edges, global_frac, std

#Todo: Change everything to var1 und var2
def fraction_in_bins_grouped(
        df_tau1, 
        df_tau2,
        frac_file: str,
        var1='pt_1',
        var2='pt_2',
        region='AR_like', 
        ar_file=None,
        grouping=None, grouping_variable=None, grouping_definition=None):
    '''
    Calculate the tau-1 fraction independently for every requested group.

    ``grouping_variable`` may either be one column name (for example
    ``"njets"``) or the two column names belonging to tau 1 and tau 2 (for
    example ``["tau_decaymode_1", "tau_decaymode_2"]``).  The return value is
    a dictionary mapping the group name to the usual
    ``(fraction, pt1_edges, pt2_edges)`` tuple.
    '''

    if grouping_variable is None or grouping_definition is None or grouping is None:
        logger.warning("Grouping variable or grouping definition or grouping is None. Calculating ungrouped fractions instead.")
        return fraction_in_bins(df_tau1, df_tau2, frac_file, var1, var2, region)

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
    all_frac = load_config(frac_file)
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
            weights_tau1 = df1["weight"] * df1[f"ff_dnn_tau1_{grouping}_raw"]
            weights_tau2 = df2["weight"] * df2[f"ff_dnn_tau2_{grouping}_raw"]
        else:
            raise ValueError(f"Unknown region: {region!r}. Expected 'AR_like' or 'AR'.")
    
        # ----- bins -----
        var2_values = np.concatenate([df1[var2].to_numpy(), df2[var2].to_numpy()])
        weights = np.concatenate([weights_tau1.to_numpy(), weights_tau2.to_numpy()])

        if region == 'AR' and ar_file is not None and var1=='pt_1' and var2=='pt_2':
            var1_bin_edges = ar_file[grouping][group_name]['pt1_edges']
            var2_bin_edges = ar_file[grouping][group_name]['pt2_edges']
        else:
            var2_bin_edges = _equal_weight_bin_edges(
                var2_values,
                weights,
                events_per_bin=5000,
            )
            var1_bin_edges = var2_bin_edges
    
        # ----- counts -----
        f1_t2, var1_edges, var2_edges = np.histogram2d(
            df1[var1],
            df1[var2],
            bins=(var1_bin_edges, var2_bin_edges),
            weights=weights_tau1,
        )
    
        t1_f2, _, _ = np.histogram2d(
            df2[var1],
            df2[var2],
            bins=(var1_bin_edges, var2_bin_edges),
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

        grouped_fractions[group_name] = fraction, var1_edges, var2_edges, global_frac, std

        if var1=='pt_1' and var2=='pt_2':
            all_frac[region][grouping][group_name] = {
                'fraction': fraction,
                'pt1_edges': var1_edges,
                'pt2_edges': var2_edges,
                'global_frac': global_frac,
                'global_std': std,
                }

    # ----- save fraction in yaml for plotting -----
    
    write_yaml_to_file(all_frac, frac_file)

    logger.info("Calculated fraction factors for group %s", group_name)

    return grouped_fractions

def fraction_in_bins_grouped_3split(
        which_frac,
        df_tau1, 
        df_tau2,
        df_tau3,
        frac_file: str,
        var1='pt_1',
        var2='pt_2',
        region='AR_like', 
        ar_file=None,
        grouping=None, grouping_variable=None, grouping_definition=None):
    '''
    Calculate the tau-1 fraction independently for every requested group.

    ``grouping_variable`` may either be one column name (for example
    ``"njets"``) or the two column names belonging to tau 1 and tau 2 (for
    example ``["tau_decaymode_1", "tau_decaymode_2"]``).  The return value is
    a dictionary mapping the group name to the usual
    ``(fraction, var1_edges, var2_edges)`` tuple.
    '''

    if grouping_variable is None or grouping_definition is None or grouping is None:
        logger.warning("Grouping variable or grouping definition or grouping is None. Calculating ungrouped fractions instead.")
        return fraction_in_bins_3split(which_frac,df_tau1, df_tau2, df_tau3, frac_file, var1, var2, region)

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

    group_tau3_masks = dict(_build_group_masks(
            np.asarray(df_tau3[grouping_var_1]), grouping_definition
        ))

    grouped_fractions = {}
    all_frac = load_config(frac_file)
    for group_name, tau1_mask in group_tau1_masks:
        tau2_mask = group_tau2_masks[group_name]
        tau3_mask = group_tau3_masks[group_name]
        if not np.any(tau1_mask) and not np.any(tau2_mask) and not np.any(tau3_mask):
            raise ValueError(f"Group {group_name!r} contains no events.")
        
        # ----- fraction in bins for grouped -----
        df1 = df_tau1.loc[tau1_mask]
        df2 = df_tau2.loc[tau2_mask]
        df3 = df_tau3.loc[tau3_mask]

        # ----- weights -----
        if region == 'AR_like':
            weights_tau1 = df1["weight_qcd"] * df1[f"ff_DR_dnn_1_{grouping}"]
            weights_tau2 = df2["weight_qcd"] * df2[f"ff_DR_dnn_2_{grouping}"]
            weights_tau3 = df3["weight_qcd"] * df3[f"ff_DR_dnn_3_{grouping}"]
        elif region == 'AR':
            weights_tau1 = df1["weight"] * df1[f"ff_dnn_1_{grouping}_raw"]
            weights_tau2 = df2["weight"] * df2[f"ff_dnn_2_{grouping}_raw"]
            weights_tau3 = df3["weight"] * df3[f"ff_dnn_3_{grouping}_raw"]
        else:
            raise ValueError(f"Unknown region: {region!r}. Expected 'AR_like' or 'AR'.")
    
        # ----- bins -----
        var2_values = np.concatenate([df1[var2].to_numpy(), df2[var2].to_numpy(), df3[var2].to_numpy()])
        weights = np.concatenate([weights_tau1.to_numpy(), weights_tau2.to_numpy(), weights_tau3.to_numpy()])

        if region == 'AR' and ar_file is not None and var1=='pt_1' and var2=='pt_2':
            var1_bin_edges = ar_file[grouping][group_name]['pt1_edges']
            var2_bin_edges = ar_file[grouping][group_name]['pt2_edges']
        else:
            var2_bin_edges = _equal_weight_bin_edges(
                var2_values,
                weights,
                events_per_bin=5000,
            )
            var1_bin_edges = var2_bin_edges
    
        # ----- counts -----
        f1_t2, var1_edges, var2_edges = np.histogram2d(
            df1[var1],
            df1[var2],
            bins=(var1_bin_edges, var2_bin_edges),
            weights=weights_tau1,
        )
    
        t1_f2, _, _ = np.histogram2d(
            df2[var1],
            df2[var2],
            bins=(var1_bin_edges, var2_bin_edges),
            weights=weights_tau2,
        )

        f1_f2, _, _ = np.histogram2d(
            df3[var1],
            df3[var2],
            bins=(var1_bin_edges, var2_bin_edges),
            weights=weights_tau3,
        )
    
        if which_frac=='tau1': numerator = f1_t2
        elif which_frac=='tau2': numerator = t1_f2
        elif which_frac=='tau1&2': numerator = f1_f2
        else: logger.error(f'which_frac is {which_frac}. Expected tau1, tau2 or tau1&2.')
        denominator = f1_t2 + t1_f2 + f1_f2
    
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

        grouped_fractions[group_name] = fraction, var1_edges, var2_edges, global_frac, std
        if var1=='pt_1' and var2=='pt_2':
            all_frac[region][grouping][group_name] = {
                'fraction': fraction,
                'pt1_edges': var1_edges,
                'pt2_edges': var2_edges,
                'global_frac': global_frac,
                'global_std': std,
                }

    # ----- save fraction in yaml for plotting -----
    
    write_yaml_to_file(all_frac, frac_file)

    logger.info("Calculated fraction factors for group %s", grouping)

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

def fractions_for_events(frame, frac, var1_edges, var2_edges, var1='pt_1', var2='pt_2', fallback=0.5):
    var1_bin = np.searchsorted(
        var1_edges, frame[var1].to_numpy(), side="right"
    ) - 1
    var2_bin = np.searchsorted(
        var2_edges, frame[var2].to_numpy(), side="right"
    ) - 1


    # Protect against values outside the histogram range.
    var1_bin = np.clip(var1_bin, 0, frac.shape[0] - 1)
    var2_bin = np.clip(var2_bin, 0, frac.shape[1] - 1)

    event_fractions = frac[var1_bin, var2_bin]

    # Choose a fallback for bins without AR-like events.
    return np.nan_to_num(event_fractions, nan=fallback)


def fraction_for_events_grouped(
        frame,
        grouped_fractions,
        grouping_variable,
        grouping_definition,
        var1='pt_1',
        var2='pt_2',
        fallback=0.5,
):
    """Look up the appropriate grouped fraction for every event."""
    if not isinstance(grouped_fractions, dict):
        raise TypeError(
            "grouped_fractions must be the dictionary returned by "
            "fraction_in_bins_grouped()."
        )

    event_fractions = np.full(len(frame), fallback, dtype=float)
    assigned = np.zeros(len(frame), dtype=bool)
    group_masks = _build_group_masks(
        np.asarray(frame[grouping_variable]), grouping_definition
    )

    for group_name, group_mask in group_masks:
        if group_name not in grouped_fractions:
            raise KeyError(f"No fraction histogram for group {group_name!r}.")
        if not np.any(group_mask):
            continue

        group_frac, var1_edges, var2_edges, _, _ = grouped_fractions[group_name]
        event_fractions[group_mask] = fractions_for_events(
            frame.loc[group_mask], group_frac, var1_edges, var2_edges, var1, var2, fallback
        )
        assigned |= group_mask

    if np.any(~assigned):
        logger.warning(
            "%d events are outside the fraction grouping definition; "
            f"using the fallback fraction {fallback}.",
            np.count_nonzero(~assigned),
        )

    return event_fractions


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

