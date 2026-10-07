import logging
#from venv import logger

import torch as t
import correctionlib as cr
import numpy as np
import pandas as pd

from .NeuralNetworks import FoldCombinedDNN, load_model, load_fold_combined_model
from .DataHandling import test_data
from classes.Fraction_factor import (
    fraction_for_events_grouped,
    fraction_in_bins,
    fraction_in_bins_grouped,
    fractions_for_events,
    fraction_in_bins_3split,
    fraction_in_bins_grouped_3split,
)
from classes.Multiclass_classification import train_fraction_classifier
from classes.Loading import write_yaml_to_file, load_config

cfg_path = load_config('/work/tapp/TauFF/NF4FF/DNN_tt/configs/config_path.yaml')
logger = logging.getLogger(__name__)

def _prepare_input_tensor(model: t.nn.Module, X_tensor: t.Tensor, df_ar) -> t.Tensor:
    """Return the correctly shaped input tensor for the given model type."""
    if isinstance(model, FoldCombinedDNN):
        event_ids = t.from_numpy(np.asarray(df_ar['event']%2, dtype=np.float32))
        return t.cat([event_ids.unsqueeze(0), X_tensor.T], dim=0)  # [1 + n_features, N]
    return X_tensor  # [N, n_features]

def _build_group_masks(values, grouping_definition):
    if grouping_definition is None:
        return None

    masks = []

    for group in grouping_definition:

        # exact value
        if len(group) == 1:

            val = group[0]

            mask = values == val
            group_name = f"{val}"

        # range
        elif len(group) == 2:

            low, high = group

            mask = (values >= low) & (values <= high)
            group_name = f"{low}_{high}"

        else:
            raise ValueError(f"Invalid group definition: {group}")

        masks.append((group_name, mask))

    return masks

def _compute_ratio(model, df_AR, training_variables):
    '''
    In SR: df_AR is either df.AR_tau1 or df.AR_tau2 or df.AR for model_tau1, model_tau2 or model respectively
    In DR: df_AR is either df.AR_like_tau1 or df.AR_like_tau2 or df.AR_like for model_tau1, model_tau2 or model respectively
    '''
    if model is None:
        return None

    X_tau = test_data(df_AR, training_variables)
    X_tensor = t.from_numpy(X_tau.X).float()
    X_prepared = _prepare_input_tensor(model, X_tensor, df_AR)
    
    with t.no_grad():
        f = model(X_prepared).cpu().numpy().flatten()

    eps = 1e-6
    f_ = np.clip(f, eps, 1 - eps)

    return f_ / (1.0 - f_)
    
def _FF_over_3(ff, tau_label):
    number = 0
    highest = 0
    for x in ff:
        if x > 3.0:
            number += 1
            highest = max(highest, x)
    print(f"Number of {tau_label} FF over 3.0: {number}, highest: {highest:.4f}")    

# ------------- dnn fake factor determinaiton -------------

# ----- tau split FF -----

def calculate_fake_factors_ungrouped(
    df,
    model_tau1: t.nn.Module = None,
    model_tau2: t.nn.Module = None,
    training_variables=None,
    DR: bool = False,
    AR_comb: bool = False
):
    if model_tau1 is None and model_tau2 is None:
        logger.error("Both model_tau1 and model_tau2 are None. No fake factors will be calculated.")
        return

    # ----- FF calculation specifics in DR or SR -----
    if DR:
        ar1, ar2 = df.AR_like_tau1, df.AR_like_tau2
    elif AR_comb:
        ar = df.AR_comb
        ar1, ar2 = ar, ar
    else:
        ar1, ar2 = df.AR_tau1, df.AR_tau2

    ratio_tau1 = _compute_ratio(model_tau1, ar1, training_variables)
    ratio_tau2 = _compute_ratio(model_tau2, ar2, training_variables)
    
    # ----- FF calculation -----
    norm_tau1 = (
        np.sum(df.data.SR_like.weight_qcd)
        / np.sum(df.data.AR_like_tau1.weight_qcd)
    )    

    norm_tau2 = (
        np.sum(df.data.SR_like.weight_qcd)
        / np.sum(df.data.AR_like_tau2.weight_qcd)
    )    

    fake_factor_tau1 = (norm_tau1 * ratio_tau1) if ratio_tau1 is not None else None
    fake_factor_tau2 = (norm_tau2 * ratio_tau2) if ratio_tau2 is not None else None

    print(f"tau1 norm = {norm_tau1:.4f}")
    print(f"tau2 norm = {norm_tau2:.4f}")

    # ----- number of FF over 3 -----
    _FF_over_3(fake_factor_tau1, "tau1")
    _FF_over_3(fake_factor_tau2, "tau2")

    # ----- clipping + output assignment -----

    if fake_factor_tau1 is None and fake_factor_tau2 is None:
        logger.error("FF for tau 1 is None or FF for tau 2 is None")

    if DR:
        ar1[f"ff_DR_unclipped_dnn_tau1"] = fake_factor_tau1
        fake_factor_tau1 = np.clip(fake_factor_tau1, 0, 3)
        ar1[f"ff_DR_dnn_tau1"] = fake_factor_tau1            

        ar2[f"ff_DR_unclipped_dnn_tau2"] = fake_factor_tau2
        fake_factor_tau2 = np.clip(fake_factor_tau2, 0, 3)
        ar2[f"ff_DR_dnn_tau2"] = fake_factor_tau2
    else:
        ar1[f"ff_unclipped_dnn_tau1"] = fake_factor_tau1
        fake_factor_tau1 = np.clip(fake_factor_tau1, 0, 3)
        ar1[f"ff_dnn_tau1_raw"] = fake_factor_tau1            

        ar2[f"ff_unclipped_dnn_tau2"] = fake_factor_tau2
        fake_factor_tau2 = np.clip(fake_factor_tau2, 0, 3)
        ar2[f"ff_dnn_tau2_raw"] = fake_factor_tau2

def calculate_fake_factors_grouped(
    df,
    model_tau1: t.nn.Module = None,
    model_tau2: t.nn.Module = None,
    training_variables=None,
    DR: bool = False,
    grouping_variable=None,
    grouping_definition=None,
    output_suffix=None,
    AR_comb = False,
    normalization_file = None
):
    if output_suffix is None or grouping_variable is None or grouping_definition is None:
        calculate_fake_factors_ungrouped(df, model_tau1=model_tau1, model_tau2=model_tau2, training_variables=training_variables, DR=DR, AR_comb=AR_comb)
        logger.warning("Grouping variable, grouping definition, or output suffix is None. Calculating ungrouped fake factors instead.")
        return

    if model_tau1 is None and model_tau2 is None:
        logger.error("Both model_tau1 and model_tau2 are None. No fake factors will be calculated.")
        return

    # ----- grouping variable handling -----
    if isinstance(grouping_variable, list):
        grouping_var_1 = grouping_variable[0]
        grouping_var_2 = grouping_variable[1]
    else:
        grouping_var_1 = grouping_variable
        grouping_var_2 = grouping_variable

    # ----- FF calculation specifics in DR or SR -----
    if DR:
        ar1, ar2 = df.AR_like_tau1, df.AR_like_tau2
    elif AR_comb:
        ar = df.AR_comb
        ar1, ar2 = ar, ar
    else:
        ar1, ar2 = df.AR_tau1, df.AR_tau2
    
    ar_tau1_group_values = np.asarray(ar1[grouping_var_1])
    ar_tau2_group_values = np.asarray(ar2[grouping_var_2])

    ratio_tau1 = _compute_ratio(model_tau1, ar1, training_variables)
    ratio_tau2 = _compute_ratio(model_tau2, ar2, training_variables)
    


    # Ar masks for each group
    group_tau1_masks = _build_group_masks(ar_tau1_group_values, grouping_definition)
    group_tau2_masks = _build_group_masks(ar_tau2_group_values, grouping_definition)
    
    # AR-like and SR-like masks for each group
    sr_tau1_masks = dict(_build_group_masks(np.asarray(df.data.SR_like[grouping_var_1]), grouping_definition))
    ar_tau1_masks = dict(_build_group_masks(np.asarray(df.data.AR_like_tau1[grouping_var_1]), grouping_definition))

    sr_tau2_masks = dict(_build_group_masks(np.asarray(df.data.SR_like[grouping_var_2]), grouping_definition))
    ar_tau2_masks = dict(_build_group_masks(np.asarray(df.data.AR_like_tau2[grouping_var_2]), grouping_definition))
    
    
    # ----- Main FF calculation -----
    fake_factor_tau1 = np.zeros_like(ratio_tau1) if ratio_tau1 is not None else None
    fake_factor_tau2 = np.zeros_like(ratio_tau2) if ratio_tau2 is not None else None

    normalization_tau1 = {}
    for group_name, ar_mask in group_tau1_masks:

        sr_tau1_mask = sr_tau1_masks[group_name]
        ar_tau1_mask = ar_tau1_masks[group_name]

        norm_tau1 = (
            np.sum(df.data.SR_like.weight_qcd[sr_tau1_mask])
            / np.sum(df.data.AR_like_tau1.weight_qcd[ar_tau1_mask])
        )
        normalization_tau1[group_name] = norm_tau1

        fake_factor_tau1[ar_mask] = (norm_tau1 * ratio_tau1[ar_mask])

        print(f"{group_name}:tau1 norm = {norm_tau1:.4f}")

    normalization_tau2 = {}
    for group_name, ar_mask in group_tau2_masks:

        sr_tau2_mask = sr_tau2_masks[group_name]
        ar_tau2_mask = ar_tau2_masks[group_name]

        norm_tau2 = (
            np.sum(df.data.SR_like.weight_qcd[sr_tau2_mask])
            / np.sum(df.data.AR_like_tau2.weight_qcd[ar_tau2_mask])
        )
        normalization_tau2[group_name] = norm_tau2

        fake_factor_tau2[ar_mask] = (norm_tau2 * ratio_tau2[ar_mask])

        print(f"{group_name}:tau2 norm = {norm_tau2:.4f}")

    
    # Save normalization values to YAML file
    if normalization_file is not None:
        norm_file = load_config(normalization_file)
        norm_file['tau1'][output_suffix] = normalization_tau1
        norm_file['tau2'][output_suffix] = normalization_tau2
        write_yaml_to_file(norm_file, normalization_file)
    
    # ----- number of FF over 3 -----
    _FF_over_3(fake_factor_tau1, "tau1")
    _FF_over_3(fake_factor_tau2, "tau2")


    # ----- clipping + output assignment -----
    suffix = f"_{output_suffix}"

    if fake_factor_tau1 is None or fake_factor_tau2 is None:
        print("FF for tau 1 is None or FF for tau 2 is None")

    if DR:
        ar1[f"ff_DR_unclipped_dnn_tau1{suffix}"] = fake_factor_tau1
        fake_factor_tau1 = np.clip(fake_factor_tau1, 0, 3)
        ar1[f"ff_DR_dnn_tau1{suffix}"] = fake_factor_tau1

        ar2[f"ff_DR_unclipped_dnn_tau2{suffix}"] = fake_factor_tau2
        fake_factor_tau2 = np.clip(fake_factor_tau2, 0, 3)
        ar2[f"ff_DR_dnn_tau2{suffix}"] = fake_factor_tau2
    else:
        ar1[f"ff_unclipped_dnn_tau1{suffix}"] = fake_factor_tau1
        fake_factor_tau1 = np.clip(fake_factor_tau1, 0, 3)
        ar1[f"ff_dnn_tau1{suffix}_raw"] = fake_factor_tau1

        ar2[f"ff_unclipped_dnn_tau2{suffix}"] = fake_factor_tau2
        fake_factor_tau2 = np.clip(fake_factor_tau2, 0, 3)
        ar2[f"ff_dnn_tau2{suffix}_raw"] = fake_factor_tau2

def calculate_fake_factor_frac(
        df,
        df1,
        df2,
        frac_file,
        grouping = None,
        grouping_variable = None,
        grouping_definition = None,
        fraction = "global",
        training_variables = None,
        train_frac=True,
):
    '''
    Applying fraction factor to FF for tau split.

    :fraction: "global" (factor 1/2), "pt_binned", or "DNN".
    DNN fractions use one classifier for tau_dm with both decay modes as
    inputs and tau_dm DR weights; njets uses a classifier per jet category.
    '''
    _df = df.copy()
    _df1 = df1.copy()
    _df2 = df2.copy()

    if grouping is None:
        ff_tau1 = "ff_dnn_tau1_raw"
        ff_tau2 = "ff_dnn_tau2_raw"
    elif grouping == 'tau_dm':
        ff_tau1 = "ff_dnn_tau1_tau_dm_raw"
        ff_tau2 = "ff_dnn_tau2_tau_dm_raw"
    elif grouping == 'njets':
        ff_tau1 = "ff_dnn_tau1_njets_raw"
        ff_tau2 = "ff_dnn_tau2_njets_raw"

    if fraction == "global":
        df1[ff_tau1.removesuffix('_raw')] = 0.5 * _df1[ff_tau1]
        df2[ff_tau2.removesuffix('_raw')] = 0.5 * _df2[ff_tau2]

    elif fraction == "pt_binned":
        suffix = "" if grouping is None else f"_{grouping}"
        if grouping is None:
            frac, pt1_edges, pt2_edges, _, _ = fraction_in_bins(df.data.AR_like_tau1, df.data.AR_like_tau2, frac_file)

            frac_tau1 = fractions_for_events(_df1, frac, pt1_edges, pt2_edges)
            frac_tau2 = fractions_for_events(_df2, frac, pt1_edges, pt2_edges)

            df1[f"fraction_pt_binned_1{suffix}"] = frac_tau1
            df2[f"fraction_pt_binned_2{suffix}"] = 1.0 - frac_tau2
            df1[ff_tau1.removesuffix('_raw')] = frac_tau1 * _df1[ff_tau1]
            df2[ff_tau2.removesuffix('_raw')] = (1.0 - frac_tau2) * _df2[ff_tau2]
            logger.info(f'Saved Fraction Factors for ungrouped')
            
        else:
            if grouping_variable is None or grouping_definition is None:
                raise ValueError("grouping_variable and grouping_definition are required for grouped fraction factors.")

            if isinstance(grouping_variable, list):
                if len(grouping_variable) != 2:
                    raise ValueError("grouping_variable must contain exactly two column names when supplied as a list.")
                grouping_var_1, grouping_var_2 = grouping_variable
            else:
                grouping_var_1 = grouping_var_2 = grouping_variable

            grouped_frac = fraction_in_bins_grouped(
                df.data.AR_like_tau1,
                df.data.AR_like_tau2,
                frac_file=frac_file,
                grouping=grouping,
                grouping_variable=grouping_variable,
                grouping_definition=grouping_definition,
            )
            frac_tau1 = fraction_for_events_grouped(
                _df1,
                grouped_frac,
                grouping_variable=grouping_var_1,
                grouping_definition=grouping_definition,
            )
            frac_tau2 = fraction_for_events_grouped(
                _df2,
                grouped_frac,
                grouping_variable=grouping_var_2,
                grouping_definition=grouping_definition,
            )
            df1[f"fraction_pt_binned_1{suffix}"] = frac_tau1
            df2[f"fraction_pt_binned_2{suffix}"] = 1.0 - frac_tau2
            target_dtype = _df1[ff_tau1].dtype
            df1[ff_tau1.removesuffix('_raw')] = (frac_tau1 * _df1[ff_tau1]).astype(target_dtype)
            target_dtype = _df2[ff_tau2].dtype
            df2[ff_tau2.removesuffix('_raw')] = (1.0 - frac_tau2) * _df2[ff_tau2].astype(target_dtype)
            logger.info("Saved Fraction Factors for grouping %s", grouping)
            return grouped_frac

    elif fraction == "DNN":
        if grouping is None or grouping == "tau_dm":
            grouping = 'ungrouped' if grouping is None else grouping
            output_dir = f"/work/tapp/TauFF/NF4FF/DNN_tt/Training_multiclass_results/split/{grouping}"
            suffix = "_tau_dm" if grouping == "tau_dm" else ""
            if grouping == "tau_dm":
                # Both classes must describe the same event categories. Selecting
                # dm_1 for class 0 and dm_2 for class 1 breaks this correspondence.
                output_dir += "/combined"
                if training_variables is None:
                    raise ValueError("training_variables are required for DNN fraction factors.")
                training_variables = list(training_variables)
                for variable in ("tau_decaymode_1", "tau_decaymode_2"):
                    if variable not in training_variables:
                        training_variables.append(variable)

            if train_frac:
                train_fraction_classifier(
                    df=df,
                    training_var=training_variables,
                    output_dir=output_dir,
                    number_of_classes=2,
                    training_masks=None,
                    ff_suffix=suffix,
                    save_file=cfg_path['DNN_output']+'/frac_split.yaml',
                    save_column = grouping,
                )

            model = load_model(output_dir, device="cpu")
            model.eval()

            for class_idx, region, raw_ff in zip(range(2), [df1, df2], [ff_tau1, ff_tau2]):
                data = test_data(region, training_variables)
                X = t.from_numpy(data.X)
                X = _prepare_input_tensor(model, X, region)

                with t.inference_mode():
                    logits = model(X).reshape(-1, 2)
                    fractions = t.softmax(logits, dim=1)
                    fraction_i = fractions[:, class_idx].cpu().numpy()

                region[f"fraction_dnn_{class_idx + 1}{suffix}"] = fraction_i
                region[raw_ff.removesuffix("_raw")] = fraction_i * region[raw_ff]
        else:
            if grouping_variable is None or grouping_definition is None:
                raise ValueError("grouping_variable and grouping_definition are required for grouped fraction factors.")

            if isinstance(grouping_variable, list):
                if len(grouping_variable) != 2:
                    raise ValueError("grouping_variable must contain exactly two column names when supplied as a list.")
                grouping_var_1, grouping_var_2 = grouping_variable
            else:
                grouping_var_1 = grouping_var_2 = grouping_variable

            regions = [df.data.AR_like_tau1, df.data.AR_like_tau2, df1, df2]
            group_variables = [grouping_var_1, grouping_var_2] * 2
            masks = [
                dict(_build_group_masks(np.asarray(region[variable]), grouping_definition))
                for region, variable in zip(regions, group_variables)
            ]

            # Each training and application event must belong to exactly one group.
            for region, region_masks in zip(regions, masks):
                coverage = np.zeros(len(region), dtype=int)
                for mask in region_masks.values():
                    coverage += mask
                if np.any(coverage != 1):
                    raise ValueError(f"Grouping {grouping!r} must cover each event exactly once.")

            event_fractions = [np.full(len(df1), np.nan, dtype=np.float32), np.full(len(df2), np.nan, dtype=np.float32),
            ]

            for group_name in masks[0]:
                output_dir = f"/work/tapp/TauFF/NF4FF/DNN_tt/Training_multiclass_results/split/{grouping}/{group_name}"
                if train_frac:
                    train_fraction_classifier(
                        df=df,
                        training_var=training_variables,
                        output_dir=output_dir,
                        number_of_classes=2,
                        training_masks=[masks[0][group_name], masks[1][group_name]],
                        ff_suffix=f"_{grouping}",
                        save_file=cfg_path['DNN_output']+'/frac_split.yaml',
                        save_column = grouping,
                        save_key=group_name,
                    )

                model = load_model(output_dir, device="cpu")
                model.eval()

                for class_idx, region in enumerate([df1, df2]):
                    mask = masks[class_idx + 2][group_name]
                    if not mask.any():
                        continue

                    selected = [_df1, _df2][class_idx].loc[mask]
                    data = test_data(selected, training_variables)
                    X = t.from_numpy(data.X).float()
                    X = _prepare_input_tensor(model, X, selected)

                    with t.inference_mode():
                        logits = model(X).reshape(-1, 2)
                        fractions = t.softmax(logits, dim=1)
                        fraction_i = fractions[:, class_idx].cpu().numpy()

                    event_fractions[class_idx][mask] = fraction_i

            if any(not np.isfinite(fractions).all() for fractions in event_fractions):
                raise ValueError(f"Missing or nonfinite fraction predictions for grouping {grouping!r}.")

            for class_idx, region, raw_ff in zip(range(2), [df1, df2], [ff_tau1, ff_tau2]):
                fraction_i = event_fractions[class_idx]
                region[f"fraction_dnn_{class_idx + 1}_{grouping}"] = fraction_i
                region[raw_ff.removesuffix("_raw")] = fraction_i * region[raw_ff] #* 0.5


# ----- tau inclusive FF -----

def calculate_fake_factors_incl_ungrouped(
    df,
    incl,
    model: t.nn.Module = None,
    training_variables=None,
    DR: bool = False
):
    if model is None:
        logger.error("model is None. No fake factors will be calculated.")
        return

    if DR:
        ratio = _compute_ratio(model, df.AR_like, training_variables)
    else:
        ratio = _compute_ratio(model, df.AR, training_variables)
 
    norm = (
        np.sum(df.data.SR_like.weight_qcd)
        / np.sum(df.data.AR_like.weight_qcd)
    )

    fake_factor = (norm * ratio)
    print(f"norm (global FF) = {norm:.4f}")

    # ----- number of FF over 3 -----
    _FF_over_3(fake_factor, f'tau incl {incl}')

    # ----- clipping + output assignment -----
    if fake_factor is None:
        logger.error("FF is None")

    if DR:
        df.AR_like[f"ff_DR_unclipped_dnn_incl_{incl}"] = fake_factor
        fake_factor = np.clip(fake_factor, 0, 3)
        df.AR_like[f"ff_DR_dnn_incl_{incl}"] = fake_factor 
    else:
        df.AR[f"ff_unclipped_dnn_incl_{incl}"] = fake_factor
        fake_factor = np.clip(fake_factor, 0, 3)
        df.AR[f"ff_dnn_incl_{incl}"] = fake_factor           
    
def calculate_fake_factors_incl_grouped(
    df,
    incl,
    model: t.nn.Module = None,
    training_variables=None,
    DR: bool = False,
    grouping_variable=None,
    grouping_definition=None,
    output_suffix=None,
):
    if output_suffix is None or grouping_variable is None or grouping_definition is None:
        calculate_fake_factors_incl_ungrouped(df, incl, model, training_variables, DR)
        logger.warning("Grouping variable, grouping definition, or output suffix is None. Calculating ungrouped fake factors instead.")
        return
    
    if model is None:
        logger.error("model is None. No fake factors will be calculated.")
        return

    if DR:
        ratio = _compute_ratio(model, df.AR_like, training_variables)
        ar_group_values = np.asarray(df.AR_like[grouping_variable])
    else:
        ratio = _compute_ratio(model, df.AR, training_variables)
        ar_group_values = np.asarray(df.AR[grouping_variable])

    group_masks = _build_group_masks(ar_group_values, grouping_definition)
    
    sr_masks = dict(
        _build_group_masks(np.asarray(df.data.SR_like[grouping_variable]), grouping_definition)
    )
    ar_masks = dict(
        _build_group_masks(np.asarray(df.data.AR_like[grouping_variable]), grouping_definition)
    )

    # ----- Main Loop -----
    fake_factor = np.zeros_like(ratio) if ratio is not None else None

    for group_name, ar_mask in group_masks:
        sr_mask = sr_masks[group_name]
        ar_mask_ = ar_masks[group_name]

        norm = (
            np.sum(df.data.SR_like.weight_qcd[sr_mask])
            / np.sum(df.data.AR_like.weight_qcd[ar_mask_])
        )

        fake_factor[ar_mask] = (norm * ratio[ar_mask])

        print(f"norm (global FF) = {norm:.4f}")
        

    # ----- number of FF over 3 -----
    _FF_over_3(fake_factor, f'tau incl {incl}')

    # ----- clipping + output assignment -----
    suffix = f"_{output_suffix}"
    if fake_factor is None:
        print("FF is None")

    if DR:
        df.AR_like[f"ff_DR_unclipped_dnn_incl_{incl}{suffix}"] = fake_factor
        fake_factor = np.clip(fake_factor, 0, 3)
        df.AR_like[f"ff_DR_dnn_incl_{incl}{suffix}"] = fake_factor           
    else:
        df.AR[f"ff_unclipped_dnn_incl_{incl}{suffix}"] = fake_factor
        fake_factor = np.clip(fake_factor, 0, 3)
        df.AR[f"ff_dnn_incl_{incl}{suffix}"] = fake_factor



# ----- tau 3 way split FF -----
def calculate_fake_factors_3split_ungrouped(
    df,
    model1: t.nn.Module = None,
    model2: t.nn.Module = None,
    model3: t.nn.Module = None,
    training_variables=None,
    DR: bool = False,
    AR_comb: bool = False,
):
    if model1 is None or model2 is None or model3 is None:
        logger.error("One or more models are None. No fake factors will be calculated.")
        return

    # DR normalization and validation retain their separate regions.
    if DR:
        ar1, ar2, ar3 = df.AR_like_1, df.AR_like_2, df.AR_like_3
    elif AR_comb:
        ar1 = ar2 = ar3 = df.AR_comb
    else:
        ar1, ar2, ar3 = df.AR_1, df.AR_2, df.AR_3

    ratio_tau1 = _compute_ratio(model1, ar1, training_variables)
    ratio_tau2 = _compute_ratio(model2, ar2, training_variables)
    ratio_tau3 = _compute_ratio(model3, ar3, training_variables)

    # ----- FF calculation -----
    norm_tau1 = (
        np.sum(df.data.SR_like.weight_qcd)
        / np.sum(df.data.AR_like_1.weight_qcd)
    )    

    norm_tau2 = (
        np.sum(df.data.SR_like.weight_qcd)
        / np.sum(df.data.AR_like_2.weight_qcd)
    )    

    norm_tau3 = (
        np.sum(df.data.SR_like.weight_qcd)
        / np.sum(df.data.AR_like_3.weight_qcd)
    )

    fake_factor_1 = (norm_tau1 * ratio_tau1) if ratio_tau1 is not None else None
    fake_factor_2 = (norm_tau2 * ratio_tau2) if ratio_tau2 is not None else None
    fake_factor_3 = (norm_tau3 * ratio_tau3) if ratio_tau3 is not None else None

    print(f"tau1 norm = {norm_tau1:.4f}")
    print(f"tau2 norm = {norm_tau2:.4f}")
    print(f"tau3 norm = {norm_tau3:.4f}")

    # ----- number of FF over 3 -----
    _FF_over_3(fake_factor_1, "tau1")
    _FF_over_3(fake_factor_2, "tau2")
    _FF_over_3(fake_factor_3, "tau1&2")

    # ----- clipping + output assignment -----

    if fake_factor_1 is None or fake_factor_2 is None or fake_factor_3 is None:
        logger.error("FF for tau 1 is None or FF for tau 2 is None or FF for tau1&2 is None")

    if DR:
        ar1[f"ff_DR_unclipped_dnn_1"] = fake_factor_1
        fake_factor_1 = np.clip(fake_factor_1, 0, 3)
        ar1[f"ff_DR_dnn_1"] = fake_factor_1

        ar2[f"ff_DR_unclipped_dnn_2"] = fake_factor_2
        fake_factor_2 = np.clip(fake_factor_2, 0, 3)
        ar2[f"ff_DR_dnn_2"] = fake_factor_2

        ar3[f"ff_DR_unclipped_dnn_3"] = fake_factor_3
        fake_factor_3 = np.clip(fake_factor_3, 0, 3)
        ar3[f"ff_DR_dnn_3"] = fake_factor_3
    else:
        ar1[f"ff_unclipped_dnn_1"] = fake_factor_1
        fake_factor_1 = np.clip(fake_factor_1, 0, 3)
        ar1[f"ff_dnn_1_raw"] = fake_factor_1

        ar2[f"ff_unclipped_dnn_2"] = fake_factor_2
        fake_factor_2 = np.clip(fake_factor_2, 0, 3)
        ar2[f"ff_dnn_2_raw"] = fake_factor_2

        ar3[f"ff_unclipped_dnn_3"] = fake_factor_3
        fake_factor_3 = np.clip(fake_factor_3, 0, 3)
        ar3[f"ff_dnn_3_raw"] = fake_factor_3

def calculate_fake_factors_grouped_3split(
    df,
    model1: t.nn.Module = None,
    model2: t.nn.Module = None,
    model3: t.nn.Module = None,
    training_variables=None,
    DR: bool = False,
    grouping_variable=None,
    grouping_definition=None,
    output_suffix=None,
    AR_comb: bool = False,
):
    if output_suffix is None or grouping_variable is None or grouping_definition is None:
        calculate_fake_factors_3split_ungrouped(
            df, model1=model1, model2=model2, model3=model3,
            training_variables=training_variables, DR=DR, AR_comb=AR_comb,
        )
        logger.warning("Grouping variable, grouping definition, or output suffix is None. Calculating ungrouped fake factors instead.")
        return

    if model1 is None or model2 is None or model3 is None:
        logger.error("One or more models are None. No fake factors will be calculated.")
        return

    # ----- grouping variable handling -----
    if isinstance(grouping_variable, list):
        grouping_var_1 = grouping_variable[0]
        grouping_var_2 = grouping_variable[1]
        grouping_var_3 = grouping_variable[2]
    else:
        grouping_var_1 = grouping_variable
        grouping_var_2 = grouping_variable
        grouping_var_3 = grouping_variable

    # DR normalization and validation retain their separate regions.
    if DR:
        ar1, ar2, ar3 = df.AR_like_1, df.AR_like_2, df.AR_like_3
    elif AR_comb:
        ar1 = ar2 = ar3 = df.AR_comb
    else:
        ar1, ar2, ar3 = df.AR_1, df.AR_2, df.AR_3

    ar1_group_values = np.asarray(ar1[grouping_var_1])
    ar2_group_values = np.asarray(ar2[grouping_var_2])
    ar3_group_values = np.asarray(ar3[grouping_var_3])

    ratio1 = _compute_ratio(model1, ar1, training_variables)
    ratio2 = _compute_ratio(model2, ar2, training_variables)
    ratio3 = _compute_ratio(model3, ar3, training_variables)

    # Ar masks for each group
    group1_masks = _build_group_masks(ar1_group_values, grouping_definition)
    group2_masks = _build_group_masks(ar2_group_values, grouping_definition)
    group3_masks = _build_group_masks(ar3_group_values, grouping_definition)

    # AR-like and SR-like masks for each group
    sr1_masks = dict(_build_group_masks(np.asarray(df.data.SR_like[grouping_var_1]), grouping_definition))
    ar1_masks = dict(_build_group_masks(np.asarray(df.data.AR_like_1[grouping_var_1]), grouping_definition))

    sr2_masks = dict(_build_group_masks(np.asarray(df.data.SR_like[grouping_var_2]), grouping_definition))
    ar2_masks = dict(_build_group_masks(np.asarray(df.data.AR_like_2[grouping_var_2]), grouping_definition))

    sr3_masks = dict(_build_group_masks(np.asarray(df.data.SR_like[grouping_var_3]), grouping_definition))
    ar3_masks = dict(_build_group_masks(np.asarray(df.data.AR_like_3[grouping_var_3]), grouping_definition))

    # ----- Main FF calculation -----
    fake_factor_1 = np.zeros_like(ratio1) if ratio1 is not None else None
    fake_factor_2 = np.zeros_like(ratio2) if ratio2 is not None else None
    fake_factor_3 = np.zeros_like(ratio3) if ratio3 is not None else None

    for group_name, ar_mask in group1_masks:

        sr1_mask = sr1_masks[group_name]
        ar1_mask = ar1_masks[group_name]

        norm1 = (
            np.sum(df.data.SR_like.weight_qcd[sr1_mask])
            / np.sum(df.data.AR_like_1.weight_qcd[ar1_mask])
        )

        fake_factor_1[ar_mask] = (norm1 * ratio1[ar_mask])

        print(f"{group_name}:tau1 norm = {norm1:.4f}")

    for group_name, ar_mask in group2_masks:

        sr2_mask = sr2_masks[group_name]
        ar2_mask = ar2_masks[group_name]

        norm2 = (
            np.sum(df.data.SR_like.weight_qcd[sr2_mask])
            / np.sum(df.data.AR_like_2.weight_qcd[ar2_mask])
        )

        fake_factor_2[ar_mask] = (norm2 * ratio2[ar_mask])

        print(f"{group_name}:tau2 norm = {norm2:.4f}")

    for group_name, ar_mask in group3_masks:
    
        sr3_mask = sr3_masks[group_name]
        ar3_mask = ar3_masks[group_name]

        norm3 = (
            np.sum(df.data.SR_like.weight_qcd[sr3_mask])
            / np.sum(df.data.AR_like_3.weight_qcd[ar3_mask])
        )

        fake_factor_3[ar_mask] = (norm3 * ratio3[ar_mask])

        print(f"{group_name}:tau3 norm = {norm3:.4f}")

    # ----- number of FF over 3 -----
    _FF_over_3(fake_factor_1, "tau1")
    _FF_over_3(fake_factor_2, "tau2")
    _FF_over_3(fake_factor_3, "tau3")


    # ----- clipping + output assignment -----
    suffix = f"_{output_suffix}"

    if fake_factor_1 is None or fake_factor_2 is None or fake_factor_3 is None:
        print("FF for tau 1 is None or FF for tau 2 is None or FF for tau 3 is None")

    if DR:
        ar1[f"ff_DR_unclipped_dnn_1{suffix}"] = fake_factor_1
        fake_factor_1 = np.clip(fake_factor_1, 0, 3)
        ar1[f"ff_DR_dnn_1{suffix}"] = fake_factor_1

        ar2[f"ff_DR_unclipped_dnn_2{suffix}"] = fake_factor_2
        fake_factor_2 = np.clip(fake_factor_2, 0, 3)
        ar2[f"ff_DR_dnn_2{suffix}"] = fake_factor_2

        ar3[f"ff_DR_unclipped_dnn_3{suffix}"] = fake_factor_3
        fake_factor_3 = np.clip(fake_factor_3, 0, 3)
        ar3[f"ff_DR_dnn_3{suffix}"] = fake_factor_3
    else:
        ar1[f"ff_unclipped_dnn_1{suffix}"] = fake_factor_1
        fake_factor_1 = np.clip(fake_factor_1, 0, 3)
        ar1[f"ff_dnn_1{suffix}_raw"] = fake_factor_1

        ar2[f"ff_unclipped_dnn_2{suffix}"] = fake_factor_2
        fake_factor_2 = np.clip(fake_factor_2, 0, 3)
        ar2[f"ff_dnn_2{suffix}_raw"] = fake_factor_2

        ar3[f"ff_unclipped_dnn_3{suffix}"] = fake_factor_3
        fake_factor_3 = np.clip(fake_factor_3, 0, 3)
        ar3[f"ff_dnn_3{suffix}_raw"] = fake_factor_3

def calculate_fake_factor_frac_3split(
        df,
        frac_file,
        grouping = None,
        grouping_variable = None,
        grouping_definition = None,
        fraction = "global",
        where_calc_frac = 'AR_like',
        training_variables = None,
        train_frac = True,
        AR_comb: bool = False,
):
    '''
    Apply fraction factors to separate regions or all three FFs on AR_comb.

    :fraction: "global" (yield fractions), "pt_binned", or "DNN".
    Grouped DNN fractions train one three-class classifier per njets category
    using weight_qcdr.
    '''
    
    if AR_comb:
        df1 = df2 = df3 = df.AR_comb
    else:
        df1, df2, df3 = df.AR_1, df.AR_2, df.AR_3

    _df = df.copy()
    _df1 = df1.copy()
    _df2 = df2.copy()
    _df3 = df3.copy()

    if grouping is None:
        ff_tau1 = "ff_dnn_1_raw"
        ff_tau2 = "ff_dnn_2_raw"
        ff_tau3 = "ff_dnn_3_raw"
    elif grouping == 'tau_dm':
        ff_tau1 = "ff_dnn_1_tau_dm_raw"
        ff_tau2 = "ff_dnn_2_tau_dm_raw"
        ff_tau3 = "ff_dnn_3_tau_dm_raw"
    elif grouping == 'njets':
        ff_tau1 = "ff_dnn_1_njets_raw"
        ff_tau2 = "ff_dnn_2_njets_raw"
        ff_tau3 = "ff_dnn_3_njets_raw"

    if fraction == "global":
        if where_calc_frac == 'AR_like':
            denominator = np.sum(_df.data.AR_like_1['weight_qcd']) + np.sum(_df.data.AR_like_2['weight_qcd']) + np.sum(_df.data.AR_like_3['weight_qcd'])
            frac1 = np.sum(_df.data.AR_like_1['weight_qcd']) / denominator
            frac2 = np.sum(_df.data.AR_like_2['weight_qcd']) / denominator
            frac3 = np.sum(_df.data.AR_like_3['weight_qcd']) / denominator
        elif where_calc_frac == 'AR':
            denominator = np.sum(_df.AR_1['weight']) + np.sum(_df.AR_2['weight']) + np.sum(_df.AR_3['weight'])
            frac1 = np.sum(_df.AR_1['weight']) / denominator
            frac2 = np.sum(_df.AR_2['weight']) / denominator
            frac3 = np.sum(_df.AR_3['weight']) / denominator
        else:
            raise ValueError(f"Invalid value for where_calc_frac: {where_calc_frac}. Must be 'AR_like' or 'AR'.")

        frac1 = 1/3
        frac2 = 1/3
        frac3 = 1/3
        df1[ff_tau1.removesuffix('_raw')] = frac1 * _df1[ff_tau1]
        df2[ff_tau2.removesuffix('_raw')] = frac2 * _df2[ff_tau2]
        df3[ff_tau3.removesuffix('_raw')] = frac3 * _df3[ff_tau3]
        logger.info(f"Saved Fraction Factors: \nfrac1 = {frac1:.4f}, frac2 = {frac2:.4f}, frac3 = {frac3:.4f}")
        
    elif fraction == "pt_binned":
        suffix = "" if grouping is None else f"_{grouping}"
        if grouping is None:
            for class_idx, (tau, dfn, _dfn, ff) in enumerate(zip(['tau1', 'tau2', 'tau1&2'], [df1, df2, df3], [_df1, _df2, _df3], [ff_tau1, ff_tau2, ff_tau3]), start=1):
                file = frac_file+f'/fractions_{tau}.yaml'
                frac, pt1_edges, pt2_edges, _, _ = fraction_in_bins_3split(which_frac=tau, df_tau1=df.data.AR_like_1, df_tau2=df.data.AR_like_2, df_tau3=df.data.AR_like_3, frac_file=file)

                frac = fractions_for_events(_dfn, frac, pt1_edges, pt2_edges, fallback=1/3)

                dfn[f"fraction_pt_binned_{class_idx}{suffix}"] = frac
                dfn[ff.removesuffix('_raw')] = frac * _dfn[ff]
                logger.info(f'Saved Fraction Factors for ungrouped {tau}')
            
        else:
            if grouping_variable is None or grouping_definition is None:
                raise ValueError("grouping_variable and grouping_definition are required for grouped fraction factors.")

            if isinstance(grouping_variable, list):
                if len(grouping_variable) != 2:
                    raise ValueError("grouping_variable must contain exactly two column names when supplied as a list.")
                grouping_var_1, grouping_var_2 = grouping_variable
            else:
                grouping_var_1 = grouping_var_2 = grouping_variable

            for class_idx, (tau, dfn, _dfn, ff) in enumerate(zip(['tau1', 'tau2', 'tau1&2'], [df1, df2, df3], [_df1, _df2, _df3], [ff_tau1, ff_tau2, ff_tau3]), start=1):
                file = frac_file+f'/fractions_{tau}.yaml'
                grouped_frac = fraction_in_bins_grouped_3split(
                    tau,
                    df.data.AR_like_1,
                    df.data.AR_like_2,
                    df.data.AR_like_3,
                    frac_file=file,
                    grouping=grouping,
                    grouping_variable=grouping_variable,
                    grouping_definition=grouping_definition,
                )
                frac = fraction_for_events_grouped(
                    _dfn,
                    grouped_frac,
                    grouping_variable=grouping_variable,
                    grouping_definition=grouping_definition,
                    fallback=1/3,
                )
                dfn[f"fraction_pt_binned_{class_idx}{suffix}"] = frac
                target_dtype = _dfn[ff].dtype
                dfn[ff.removesuffix('_raw')] = (frac * _dfn[ff]).astype(target_dtype)
                logger.info(f"Saved Fraction Factors for grouping {grouping} for {tau}")
                #return grouped_frac

    elif fraction == "DNN":
        if grouping is None:
            output_dir = "/work/tapp/TauFF/NF4FF/DNN_tt/Training_multiclass_results/3split/ungrouped"
            if train_frac:
                train_fraction_classifier(df=df, training_var=training_variables, output_dir=output_dir, save_file=cfg_path['DNN_output']+'/frac_3split.yaml')

            model = load_model(output_dir, device="cpu")
            model.eval()

            for class_idx, region, raw_ff in zip(range(3), [df1, df2, df3], [ff_tau1, ff_tau2, ff_tau3]):
                data = test_data(region, training_variables)
                X = t.from_numpy(data.X)
                X = _prepare_input_tensor(model, X, region)

                with t.inference_mode():
                    logits = model(X).reshape(-1, 3)
                    fractions = t.softmax(logits, dim=1)
                    fraction_i = fractions[:, class_idx].cpu().numpy()

                region[f"fraction_dnn_{class_idx + 1}"] = fraction_i
                region[raw_ff.removesuffix("_raw")] = fraction_i * region[raw_ff]

        elif grouping == "njets":
            if grouping_variable != "njets" or grouping_definition is None:
                raise ValueError("grouping_variable='njets' and grouping_definition are required for njets fraction factors.")
            if training_variables is None:
                raise ValueError("training_variables are required for DNN fraction factors.")
            if where_calc_frac != "AR_like":
                raise ValueError("Grouped DNN fractions are trained in AR_like regions.")

            training_regions = [df.data.AR_like_1, df.data.AR_like_2, df.data.AR_like_3]
            application_regions = [df1, df2, df3]
            application_frames = [_df1, _df2, _df3]
            masks = [
                dict(_build_group_masks(np.asarray(region["njets"]), grouping_definition))
                for region in training_regions + application_regions
            ]

            for region, region_masks in zip(training_regions + application_regions, masks):
                coverage = np.zeros(len(region), dtype=int)
                for mask in region_masks.values():
                    coverage += mask
                if np.any(coverage != 1):
                    raise ValueError("Grouping 'njets' must cover each event exactly once.")

            event_fractions = [
                np.full(len(region), np.nan, dtype=np.float32)
                for region in application_regions
            ]

            for group_name in masks[0]:
                # Keep three-class checkpoints separate from two-class models.
                output_dir = (
                    f"/work/tapp/TauFF/NF4FF/DNN_tt/Training_multiclass_results/3split/njets/{group_name}"
                )
                if train_frac:
                    train_fraction_classifier(
                        df=df,
                        training_var=training_variables,
                        output_dir=output_dir,
                        number_of_classes=3,
                        training_masks=[region_masks[group_name] for region_masks in masks[:3]],
                        ff_suffix="_njets",
                        save_file=cfg_path['DNN_output']+'/frac_3split.yaml',
                        save_column = grouping,
                        save_key=group_name,
                    )

                model = load_model(output_dir, device="cpu")
                model.eval()

                for class_idx, frame in enumerate(application_frames):
                    mask = masks[class_idx + 3][group_name]
                    if not mask.any():
                        continue

                    selected = frame.loc[mask]
                    data = test_data(selected, training_variables)
                    X = t.from_numpy(data.X).float()
                    X = _prepare_input_tensor(model, X, selected)

                    with t.inference_mode():
                        logits = model(X).reshape(-1, 3)
                        fractions = t.softmax(logits, dim=1)
                        fraction_i = fractions[:, class_idx].cpu().numpy()

                    event_fractions[class_idx][mask] = fraction_i

            if any(not np.isfinite(fractions).all() for fractions in event_fractions):
                raise ValueError("Missing or nonfinite three-split fraction predictions for grouping 'njets'.")

            for class_idx, region, raw_ff in zip(range(3), application_regions, [ff_tau1, ff_tau2, ff_tau3]):
                fraction_i = event_fractions[class_idx]
                region[f"fraction_dnn_{class_idx + 1}_njets"] = fraction_i
                region[raw_ff.removesuffix("_raw")] = fraction_i * region[raw_ff] #* 0.5
        else:
            raise NotImplementedError(f"Grouped three-split DNN fractions are not implemented for {grouping!r}.")


# -------------- classic fake factor determination -------------
def calculate_fake_factor_classic(
        df,
        ):
    
    
    _df1 = df.AR_tau1.copy()
    _df2 = df.AR_tau2.copy()
    ff = cr.CorrectionSet.from_file('/work/tapp/TauFF/ClassicFF/TauFakeFactors/workdir/classicFF_2018_tt/2018/fake_factors_tt.json.gz')
    corr = cr.CorrectionSet.from_file('/work/tapp/TauFF/ClassicFF/TauFakeFactors/workdir/classicFF_2018_tt/2018/FF_corrections_tt_chain_DR_SR.json.gz')
    
    frac1 = ff['process_fractions']
    frac2 = ff['process_fractions_subleading']


    ff_tau1 = ff['QCD_fake_factors']
    ff_tau2 = ff['QCD_subleading_fake_factors']

    # ----- raw FF -----
    _df1["tau1_classic_ff"] = ff_tau1.evaluate(
        _df1.pt_1.values,
        _df1.njets.values,
        "nominal",
    )
    _df2['tau2_classic_ff'] = ff_tau2.evaluate(
        _df2.pt_2.values,
        _df2.njets.values,
        "nominal",
    )

    # ----- corrected FF -----
    _df1["tau1_corrected_classic_ff"] = _df1["tau1_classic_ff"] * evaluate_compound_ff_correction(
        corr,
        "QCD_compound_correction",
        _df1,
    ) * corr["QCD_DR_SR_correction"].evaluate(
        _df1.pt_tt,
        _df1.njets,
        "nominal",
    )

    _df2["tau2_corrected_classic_ff"] = _df2["tau2_classic_ff"] * evaluate_compound_ff_correction(
        corr,
        "QCD_subleading_compound_correction",
        _df2,
    ) * corr["QCD_subleading_DR_SR_correction"].evaluate(
        _df2.pt_tt,
        _df2.njets,
        "nominal",
    )
    

    _df1['process_fraction_tau1'] = frac1.evaluate(
        'QCD',
        _df1.m_vis.values,
        _df1.njets.values,
        'nominal'
    )

    _df2['process_fraction_tau2'] = frac2.evaluate(
        'QCD',
        _df2.m_vis.values,
        _df2.njets.values,
        'nominal'
    )

    #_df['corrected_ff'] = _df['process_fraction_wjets'] * _df['wjets_corrected_classic_ff'] + _df['process_fraction_qcd'] * _df['qcd_corrected_classic_ff'] + _df['process_fraction_ttbar'] * _df['ttbar_corrected_classic_ff']
    _df1['classic_ff_tau1'] = _df1['process_fraction_tau1'] * _df1['tau1_classic_ff']
    _df2['classic_ff_tau2'] = _df2['process_fraction_tau2'] * _df2['tau2_classic_ff']

    _df1['corrected_classic_ff_tau1'] = _df1['process_fraction_tau1'] * _df1['tau1_corrected_classic_ff']
    _df2['corrected_classic_ff_tau2'] = _df2['process_fraction_tau2'] * _df2['tau2_corrected_classic_ff']

    df.AR_tau1[f'ff_classic_tau1'] = _df1['classic_ff_tau1'] * 0.5
    df.AR_tau2[f'ff_classic_tau2'] = _df2['classic_ff_tau2'] * 0.5

    df.AR_tau1[f'ff_classic_tau1_corr'] = _df1['corrected_classic_ff_tau1'] * 0.5
    df.AR_tau2[f'ff_classic_tau2_corr'] = _df2['corrected_classic_ff_tau2'] * 0.5
    

def evaluate_compound_ff_correction(correction_set, compound_name: str, df: pd.DataFrame) -> np.ndarray:
    compound_correction = correction_set.compound[compound_name]
    # variables in which non closure corrections were done
    expected_inputs = [input_spec.name for input_spec in compound_correction.inputs]

    # provide connection between json and df
    input_values = {
        'tau_decaymode_1': df.tau_decaymode_1,
        'tau_decaymode_2': df.tau_decaymode_2,
        'eta_1': df.eta_1,
        'eta_2': df.eta_2,
        'jeta_1': df.jeta_1,
        'jeta_2': df.jeta_2,
        'jpt_1': df.jpt_1,
        'jpt_2': df.jpt_2,
        'deltaR_ditaupair': df.deltaR_ditaupair,
        'deltaR_1j1': df.deltaR_1j1,
        'deltaR_12j1': df.deltaR_12j1,
        'pt_ttjj': df.pt_ttjj,
        'mass_1': df.mass_1,
        'mass_2': df.mass_2,
        'mt_tot': df.mt_tot,
        'm_vis': df.m_vis,
        'iso_1': df.iso_1,
        'iso_2': df.iso_2,
        'njets': df.njets,
        'syst': 'nominal',
    }
    missing_inputs = [name for name in expected_inputs if name not in input_values]
    if missing_inputs:
        raise KeyError(f'Missing input mapping for correction {compound_name}: {missing_inputs}')

    #order the inputs in the same order as the non closure was done
    ordered_inputs = [input_values[name] for name in expected_inputs]
    return compound_correction.evaluate(*ordered_inputs)

def calculate_fake_factor_classic_jvsg(
        df,
        short=None,
        ):
    
    if short=='jv':
        _df1 = df.AR_tau1_jvoss.copy()
        _df2 = df.AR_tau2_jvoss.copy()

        ff = cr.CorrectionSet.from_file('/work/jvoss/KingMaker_sda/CROWN/analysis_configurations/tau/payloads/fake_factors/sm/2018_v3/fake_factors_tt.json.gz')
        corr = cr.CorrectionSet.from_file('/work/jvoss/KingMaker_sda/CROWN/analysis_configurations/tau/payloads/fake_factors/sm/2018_v3/FF_corrections_tt.json.gz')
    elif short=='sg':
        _df1 = df.AR_tau1_sgiappic.copy()
        _df2 = df.AR_tau2_sgiappic.copy()

        ff = cr.CorrectionSet.from_file('/work/sgiappic/KingMaker/CROWN/analysis_configurations/tau/payloads/fake_factors/sm/2024/fake_factors_tt.json.gz')
        corr = cr.CorrectionSet.from_file('/work/sgiappic/KingMaker/CROWN/analysis_configurations/tau/payloads/fake_factors/sm/2024/FF_corrections_tt.json.gz')
    else:
        _df1 = df.AR_tau1.copy()
        _df2 = df.AR_tau2.copy()
        ff = cr.CorrectionSet.from_file('/work/tapp/TauFF/ClassicFF/TauFakeFactors/workdir/classicFF_2018_tt/2018/fake_factors_tt.json.gz')
        corr = cr.CorrectionSet.from_file('/work/tapp/TauFF/ClassicFF/TauFakeFactors/workdir/classicFF_2018_tt/2018/FF_corrections_tt_chain_DR_SR.json.gz')
    
    frac1 = ff['process_fractions']
    frac2 = ff['process_fractions_subleading']


    ff_tau1 = ff['QCD_fake_factors']
    ff_tau2 = ff['QCD_subleading_fake_factors']



    _df1["tau1_classic_ff"] = ff_tau1.evaluate(
        _df1.pt_1.values,
        _df1.njets.values,
        "nominal",
    )


    _df2['tau2_classic_ff'] = ff_tau2.evaluate(
        _df2.pt_2.values,
        _df2.njets.values,
        "nominal",
    )

    if short=='sg':
        _df1["tau1_corrected_classic_ff"] = _df1["tau1_classic_ff"] * evaluate_compound_ff_correction_jvsg(
            corr,
            "QCD_compound_correction",
            _df1,
            short=short,
        ) * corr["QCD_DR_SR_correction"].evaluate(
            _df1.pt_tt,
            _df1.njets,
            "nominal",
        )

        _df2["tau2_corrected_classic_ff"] = _df2["tau2_classic_ff"] * evaluate_compound_ff_correction_jvsg(
            corr,
            "QCD_subleading_compound_correction",
            _df2,
            short=short,
        ) * corr["QCD_subleading_DR_SR_correction"].evaluate(
            _df2.pt_tt,
            _df2.njets,
            "nominal",
        )
    elif short=='jv':
        _df1["tau1_corrected_classic_ff"] = _df1["tau1_classic_ff"] * evaluate_compound_ff_correction_jvsg(
            corr,
            "QCD_compound_correction",
            _df1,
            short=short,
        ) * corr["QCD_DR_SR_correction"].evaluate(
            _df1.pt_tautau,
            _df1.njets,
            "nominal",
        )

        _df2["tau2_corrected_classic_ff"] = _df2["tau2_classic_ff"] * evaluate_compound_ff_correction_jvsg(
            corr,
            "QCD_subleading_compound_correction",
            _df2,
            short=short,
        ) * corr["QCD_subleading_DR_SR_correction"].evaluate(
            _df2.pt_tautau,
            _df2.njets,
            "nominal",
        )

    _df1['process_fraction_tau1'] = frac1.evaluate(
        'QCD',
        _df1.m_vis.values,
        _df1.njets.values,
        'nominal'
    )

    _df2['process_fraction_tau2'] = frac2.evaluate(
        'QCD',
        _df2.m_vis.values,
        _df2.njets.values,
        'nominal'
    )

    #_df['corrected_ff'] = _df['process_fraction_wjets'] * _df['wjets_corrected_classic_ff'] + _df['process_fraction_qcd'] * _df['qcd_corrected_classic_ff'] + _df['process_fraction_ttbar'] * _df['ttbar_corrected_classic_ff']
    _df1['classic_ff_tau1'] = _df1['process_fraction_tau1'] * _df1['tau1_classic_ff']
    _df2['classic_ff_tau2'] = _df2['process_fraction_tau2'] * _df2['tau2_classic_ff']

    _df1['corrected_classic_ff_tau1'] = _df1['process_fraction_tau1'] * _df1['tau1_corrected_classic_ff']
    _df2['corrected_classic_ff_tau2'] = _df2['process_fraction_tau2'] * _df2['tau2_corrected_classic_ff']

    if short=='jv':
        df.AR_tau1_jvoss[f'ff_classic_tau1_{short}'] = _df1['classic_ff_tau1']
        df.AR_tau2_jvoss[f'ff_classic_tau2_{short}'] = _df2['classic_ff_tau2']

        df.AR_tau1_jvoss[f'ff_corr_classic_tau1_{short}'] = _df1['corrected_classic_ff_tau1']
        df.AR_tau2_jvoss[f'ff_corr_classic_tau2_{short}'] = _df2['corrected_classic_ff_tau2']
    elif short=='sg':
        df.AR_tau1_sgiappic[f'ff_classic_tau1_{short}'] = _df1['classic_ff_tau1']
        df.AR_tau2_sgiappic[f'ff_classic_tau2_{short}'] = _df2['classic_ff_tau2']

        df.AR_tau1_sgiappic[f'ff_corr_classic_tau1_{short}'] = _df1['corrected_classic_ff_tau1']
        df.AR_tau2_sgiappic[f'ff_corr_classic_tau2_{short}'] = _df2['corrected_classic_ff_tau2']
    else:
        print(f'short = {short} is not implmented. Use either jv or sg')

def evaluate_compound_ff_correction_jvsg(correction_set, compound_name: str, df: pd.DataFrame, short) -> np.ndarray:
    compound_correction = correction_set.compound[compound_name]
    expected_inputs = [input_spec.name for input_spec in compound_correction.inputs]

    if short == 'sg':
        input_values = {
            'tau_decaymode_1': df.tau_decaymode_1,
            'tau_decaymode_2': df.tau_decaymode_2,
            'eta_1': df.eta_1,
            'eta_2': df.eta_2,
            'jeta_1': df.jeta_1,
            'jeta_2': df.jeta_2,
            'jpt_1': df.jpt_1,
            'jpt_2': df.jpt_2,
            'deltaR_ditaupair': df.deltaR_ditaupair,
            'deltaR_1j1': df.deltaR_1j1,
            'deltaR_12j1': df.deltaR_12j1,
            'pt_ttjj': df.pt_ttjj,
            'mass_1': df.mass_1,
            'mass_2': df.mass_2,
            'mt_tot': df.mt_tot,
            'm_vis': df.m_vis,
            'iso_1': df.iso_1,
            'njets': df.njets,
            'syst': 'nominal',
        }
    elif short == 'jv':
        input_values = {
            'tau_decaymode_1': df.tau_decaymode_1,
            'tau_decaymode_2': df.tau_decaymode_2,
            'jeta_1': df.jeta_1,
            'jeta_2': df.jeta_2,
            'jpt_1': df.jpt_1,
            'jpt_2': df.jpt_2,
            'met': df.met,
            'deltaR_ditaupair': df.deltaR_ditaupair,
            'mass_1': df.mass_1,
            'mass_2': df.mass_2,
            'mt_tot': df.mt_tot,
            'm_vis': df.m_vis,
            'iso_1': df.iso_1,
            'njets': df.njets,
            'pt_1': df.pt_1,
            'pt_2': df.pt_2,
            'pt_vis': df.pt_vis,
            'syst': 'nominal',
        }

    missing_inputs = [name for name in expected_inputs if name not in input_values]
    if missing_inputs:
        raise KeyError(f'Missing input mapping for correction {compound_name}: {missing_inputs}')

    ordered_inputs = [input_values[name] for name in expected_inputs]
    return compound_correction.evaluate(*ordered_inputs)
