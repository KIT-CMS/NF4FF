import logging
#from venv import logger

import torch as t
import correctionlib as cr
import numpy as np
import pandas as pd

from .NeuralNetworks import FoldCombinedDNN
from .DataHandling import test_data
from classes.Fraction_factor import (
    fraction_for_events_grouped,
    fraction_in_bins,
    fraction_in_bins_grouped,
    fractions_for_events,
)
from classes.Loading import write_yaml_to_file, load_config


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
):
    if model_tau1 is None and model_tau2 is None:
        logger.error("Both model_tau1 and model_tau2 are None. No fake factors will be calculated.")
        return

    # ----- FF calculation specifics in DR or SR -----
    if DR:
        ratio_tau1 = _compute_ratio(model_tau1, df.AR_like_tau1, training_variables)
        ratio_tau2 = _compute_ratio(model_tau2, df.AR_like_tau2, training_variables)
    else:
        ratio_tau1 = _compute_ratio(model_tau1, df.AR_tau1, training_variables)
        ratio_tau2 = _compute_ratio(model_tau2, df.AR_tau2, training_variables) 

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
        df.AR_like_tau1[f"ff_DR_unclipped_dnn_tau1"] = fake_factor_tau1
        fake_factor_tau1 = np.clip(fake_factor_tau1, 0, 3)
        df.AR_like_tau1[f"ff_DR_dnn_tau1"] = fake_factor_tau1            

        df.AR_like_tau2[f"ff_DR_unclipped_dnn_tau2"] = fake_factor_tau2
        fake_factor_tau2 = np.clip(fake_factor_tau2, 0, 3)
        df.AR_like_tau2[f"ff_DR_dnn_tau2"] = fake_factor_tau2
    else:
        df.AR_tau1[f"ff_unclipped_dnn_tau1"] = fake_factor_tau1
        fake_factor_tau1 = np.clip(fake_factor_tau1, 0, 3)
        df.AR_tau1[f"ff_dnn_tau1"] = fake_factor_tau1            

        df.AR_tau2[f"ff_unclipped_dnn_tau2"] = fake_factor_tau2
        fake_factor_tau2 = np.clip(fake_factor_tau2, 0, 3)
        df.AR_tau2[f"ff_dnn_tau2"] = fake_factor_tau2

def calculate_fake_factors_grouped(
    df,
    model_tau1: t.nn.Module = None,
    model_tau2: t.nn.Module = None,
    training_variables=None,
    DR: bool = False,
    grouping_variable=None,
    grouping_definition=None,
    output_suffix=None,
):
    if output_suffix is None or grouping_variable is None or grouping_definition is None:
        calculate_fake_factors_ungrouped(df, model_tau1=model_tau1, model_tau2=model_tau2, training_variables=training_variables, DR=DR)
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
        ar_tau1_group_values = np.asarray(df.AR_like_tau1[grouping_var_1])
        ar_tau2_group_values = np.asarray(df.AR_like_tau2[grouping_var_2])

        ratio_tau1 = _compute_ratio(model_tau1, df.AR_like_tau1, training_variables)
        ratio_tau2 = _compute_ratio(model_tau2, df.AR_like_tau2, training_variables)
    else:
        ar_tau1_group_values = np.asarray(df.AR_tau1[grouping_var_1])
        ar_tau2_group_values = np.asarray(df.AR_tau2[grouping_var_2])

        ratio_tau1 = _compute_ratio(model_tau1, df.AR_tau1, training_variables)
        ratio_tau2 = _compute_ratio(model_tau2, df.AR_tau2, training_variables)


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

    for group_name, ar_mask in group_tau1_masks:

        sr_tau1_mask = sr_tau1_masks[group_name]
        ar_tau1_mask = ar_tau1_masks[group_name]

        norm_tau1 = (
            np.sum(df.data.SR_like.weight_qcd[sr_tau1_mask])
            / np.sum(df.data.AR_like_tau1.weight_qcd[ar_tau1_mask])
        )

        fake_factor_tau1[ar_mask] = (norm_tau1 * ratio_tau1[ar_mask])

        print(f"{group_name}:tau1 norm = {norm_tau1:.4f}")

    for group_name, ar_mask in group_tau2_masks:

        sr_tau2_mask = sr_tau2_masks[group_name]
        ar_tau2_mask = ar_tau2_masks[group_name]

        norm_tau2 = (
            np.sum(df.data.SR_like.weight_qcd[sr_tau2_mask])
            / np.sum(df.data.AR_like_tau2.weight_qcd[ar_tau2_mask])
        )

        fake_factor_tau2[ar_mask] = (norm_tau2 * ratio_tau2[ar_mask])

        print(f"{group_name}:tau2 norm = {norm_tau2:.4f}")

    # ----- number of FF over 3 -----
    _FF_over_3(fake_factor_tau1, "tau1")
    _FF_over_3(fake_factor_tau2, "tau2")


    # ----- clipping + output assignment -----
    suffix = f"_{output_suffix}"

    if fake_factor_tau1 is None or fake_factor_tau2 is None:
        print("FF for tau 1 is None or FF for tau 2 is None")

    if DR:
        df.AR_like_tau1[f"ff_DR_unclipped_dnn_tau1{suffix}"] = fake_factor_tau1
        fake_factor_tau1 = np.clip(fake_factor_tau1, 0, 3)
        df.AR_like_tau1[f"ff_DR_dnn_tau1{suffix}"] = fake_factor_tau1

        df.AR_like_tau2[f"ff_DR_unclipped_dnn_tau2{suffix}"] = fake_factor_tau2
        fake_factor_tau2 = np.clip(fake_factor_tau2, 0, 3)
        df.AR_like_tau2[f"ff_DR_dnn_tau2{suffix}"] = fake_factor_tau2
    else:
        df.AR_tau1[f"ff_unclipped_dnn_tau1{suffix}"] = fake_factor_tau1
        fake_factor_tau1 = np.clip(fake_factor_tau1, 0, 3)
        df.AR_tau1[f"ff_dnn_tau1{suffix}"] = fake_factor_tau1

        df.AR_tau2[f"ff_unclipped_dnn_tau2{suffix}"] = fake_factor_tau2
        fake_factor_tau2 = np.clip(fake_factor_tau2, 0, 3)
        df.AR_tau2[f"ff_dnn_tau2{suffix}"] = fake_factor_tau2

#used todo: add fraction I calculate
def calculate_fake_factor_frac(
        df,
        df1,
        df2,
        frac_file,
        grouping = None,
        grouping_variable = None,
        grouping_definition = None,
        fraction = "global",
):
    '''
    Applying fraction factor to FF for tau split.

    :fraction: either "global" (every FF gets a factorized with 1/2) or "pt_bins"
    '''
    _df = df.copy()
    _df1 = df1.copy()
    _df2 = df2.copy()

    if grouping is None:
        ff_tau1 = "ff_dnn_tau1"
        ff_tau2 = "ff_dnn_tau2"
    elif grouping == 'tau_dm':
        ff_tau1 = "ff_dnn_tau1_tau_dm"
        ff_tau2 = "ff_dnn_tau2_tau_dm"
    elif grouping == 'njets':
        ff_tau1 = "ff_dnn_tau1_njets"
        ff_tau2 = "ff_dnn_tau2_njets"

    if fraction == "global":
        df1[ff_tau1] = 0.5 * _df1[ff_tau1]
        df2[ff_tau2] = 0.5 * _df2[ff_tau2]

    elif fraction == "pt_binned":
        if grouping is None:
            frac, pt1_edges, pt2_edges = fraction_in_bins(df.data.AR_like_tau1, df.data.AR_like_tau2, frac_file)

            frac_tau1 = fractions_for_events(_df1, frac, pt1_edges, pt2_edges)
            frac_tau2 = fractions_for_events(_df2, frac, pt1_edges, pt2_edges)

            df1[ff_tau1] = frac_tau1 * _df1[ff_tau1]
            df2[ff_tau2] = (1.0 - frac_tau2) * _df2[ff_tau2]
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
            target_dtype = _df1[ff_tau1].dtype
            df1[ff_tau1] = (frac_tau1 * _df1[ff_tau1]).astype(target_dtype)
            target_dtype = _df2[ff_tau2].dtype
            df2[ff_tau2] = (1.0 - frac_tau2) * _df2[ff_tau2].astype(target_dtype)
            logger.info("Saved Fraction Factors for grouping %s", grouping)
            return grouped_frac
    


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
):
    if model1 is None or model2 is None or model3 is None:
        logger.error("One or more models are None. No fake factors will be calculated.")
        return

    # ----- FF calculation specifics in DR or SR -----
    if DR:
        ratio_tau1 = _compute_ratio(model1, df.AR_like_1, training_variables)
        ratio_tau2 = _compute_ratio(model2, df.AR_like_2, training_variables)
        ratio_tau3 = _compute_ratio(model3, df.AR_like_3, training_variables)
    else:
        ratio_tau1 = _compute_ratio(model1, df.AR_1, training_variables)
        ratio_tau2 = _compute_ratio(model2, df.AR_2, training_variables)
        ratio_tau3 = _compute_ratio(model3, df.AR_3, training_variables)

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
    _FF_over_3(fake_factor_3, "tau1&tau2")

    # ----- clipping + output assignment -----

    if fake_factor_1 is None or fake_factor_2 is None or fake_factor_3 is None:
        logger.error("FF for tau 1 is None or FF for tau 2 is None or FF for tau1&tau2 is None")

    if DR:
        df.AR_like_1[f"ff_DR_unclipped_dnn_1"] = fake_factor_1
        fake_factor_1 = np.clip(fake_factor_1, 0, 3)
        df.AR_like_1[f"ff_DR_dnn_1"] = fake_factor_1            

        df.AR_like_2[f"ff_DR_unclipped_dnn_2"] = fake_factor_2
        fake_factor_2 = np.clip(fake_factor_2, 0, 3)
        df.AR_like_2[f"ff_DR_dnn_2"] = fake_factor_2

        df.AR_like_3[f"ff_DR_unclipped_dnn_3"] = fake_factor_3
        fake_factor_3 = np.clip(fake_factor_3, 0, 3)
        df.AR_like_3[f"ff_DR_dnn_3"] = fake_factor_3
    else:
        df.AR_1[f"ff_unclipped_dnn_1"] = fake_factor_1
        fake_factor_1 = np.clip(fake_factor_1, 0, 3)
        df.AR_1[f"ff_dnn_1"] = fake_factor_1            

        df.AR_2[f"ff_unclipped_dnn_2"] = fake_factor_2
        fake_factor_2 = np.clip(fake_factor_2, 0, 3)
        df.AR_2[f"ff_dnn_2"] = fake_factor_2

        df.AR_3[f"ff_unclipped_dnn_3"] = fake_factor_3
        fake_factor_3 = np.clip(fake_factor_3, 0, 3)
        df.AR_3[f"ff_dnn_3"] = fake_factor_3
# -------------- classic fake factor determination -------------

def calculate_fake_factor_classic(
        df,
        short,
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
        print(f'short = {short} is not implmented. Use either jv or sg')
    
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
        _df1["tau1_corrected_classic_ff"] = _df1["tau1_classic_ff"] * evaluate_compound_ff_correction(
            corr,
            "QCD_compound_correction",
            _df1,
            short=short,
        ) * corr["QCD_DR_SR_correction"].evaluate(
            _df1.pt_tt,
            _df1.njets,
            "nominal",
        )

        _df2["tau2_corrected_classic_ff"] = _df2["tau2_classic_ff"] * evaluate_compound_ff_correction(
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
        _df1["tau1_corrected_classic_ff"] = _df1["tau1_classic_ff"] * evaluate_compound_ff_correction(
            corr,
            "QCD_compound_correction",
            _df1,
            short=short,
        ) * corr["QCD_DR_SR_correction"].evaluate(
            _df1.pt_tautau,
            _df1.njets,
            "nominal",
        )

        _df2["tau2_corrected_classic_ff"] = _df2["tau2_classic_ff"] * evaluate_compound_ff_correction(
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

def evaluate_compound_ff_correction(correction_set, compound_name: str, df: pd.DataFrame, short) -> np.ndarray:
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
