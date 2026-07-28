import time

import torch as t
import correctionlib as cr
import numpy as np
import pandas as pd

from .NeuralNetworks import FoldCombinedDNN
from .DataHandling import test_data





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


# ------------- dnn fake factor determinaiton -------------
#used
def calculate_fake_factors(
    df,
    model_tau1: t.nn.Module = None,
    model_tau2: t.nn.Module = None,
    training_variables=None,
    grouping_variable=None,
    grouping_definition=None,
    output_suffix=None,
):

    X_tau1 = test_data(df.AR_tau1, training_variables)
    X_tau1_tensor = t.from_numpy(X_tau1.X).float()

    X_tau2 = test_data(df.AR_tau2, training_variables)
    X_tau2_tensor = t.from_numpy(X_tau2.X).float()

    eps = 1e-6
    suffix = f"_{output_suffix}" if output_suffix else ""

    if isinstance(grouping_variable, list):
        grouping_var_1 = grouping_variable[0]
        grouping_var_2 = grouping_variable[1]
    elif grouping_variable is not None:
        grouping_var_1 = grouping_variable
        grouping_var_2 = grouping_variable

    # ------------------------------------------------------------------
    # Helper to run inference only if model exists
    # ------------------------------------------------------------------
    def _compute_ratio(model_tau1, model_tau2):
        if model_tau1 is None or model_tau2 is None:
            return None, None

        X_tau1_prepared = _prepare_input_tensor(model_tau1, X_tau1_tensor, df.AR_tau1)
        X_tau2_prepared = _prepare_input_tensor(model_tau2, X_tau2_tensor, df.AR_tau2)

        with t.no_grad():
            f_tau1 = model_tau1(X_tau1_prepared).cpu().numpy().flatten()
            f_tau2 = model_tau2(X_tau2_prepared).cpu().numpy().flatten()

        f_tau1_ = np.clip(f_tau1, eps, 1 - eps)
        f_tau2_ = np.clip(f_tau2, eps, 1 - eps)

        return f_tau1_ / (1.0 - f_tau1_), f_tau2_ / (1.0 - f_tau2_)

    ratio_tau1, ratio_tau2 = _compute_ratio(model_tau1, model_tau2)

    # ------------------------------------------------------------------
    # Allocate outputs only for existing models
    # ------------------------------------------------------------------
    fake_factor_tau1 = (
        np.zeros_like(ratio_tau1) if ratio_tau1 is not None else None
    )

    fake_factor_tau2 = (
        np.zeros_like(ratio_tau2) if ratio_tau2 is not None else None
    )

    if grouping_variable is None:
        ar_tau1_group_values = np.asarray(df.AR_tau1)
        ar_tau2_group_values = np.asarray(df.AR_tau2)
    else:
        ar_tau1_group_values = np.asarray(df.AR_tau1[grouping_var_1])
        ar_tau2_group_values = np.asarray(df.AR_tau2[grouping_var_2])

    
    group_tau1_masks = _build_group_masks(
        ar_tau1_group_values,
        grouping_definition,
    )

    group_tau2_masks = _build_group_masks(
        ar_tau2_group_values,
        grouping_definition,
    )


    # ------------------------------------------------------------------
    # Build masks once outside the loop
    # ------------------------------------------------------------------
    
    if model_tau1 is not None and grouping_definition is not None:
        sr_tau1_masks = dict(
            _build_group_masks(
                np.asarray(df.data.SR_like[grouping_var_1]),
                grouping_definition,
            )
        )

        ar_tau1_masks = dict(
            _build_group_masks(
                np.asarray(df.data.AR_like_tau1[grouping_var_1]),
                grouping_definition,
            )
        )

    if model_tau2 is not None and grouping_definition is not None:
        sr_tau2_masks = dict(
            _build_group_masks(
                np.asarray(df.data.SR_like[grouping_var_2]),
                grouping_definition,
            )
        )

        ar_tau2_masks = dict(
            _build_group_masks(
                np.asarray(df.data.AR_like_tau2[grouping_var_2]),
                grouping_definition,
            )
        )
    
    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    if grouping_variable is None and model_tau1 is not None and model_tau2 is not None:

        norm_tau1 = (
            np.sum(df.data.SR_like.weight)
            / np.sum(df.data.AR_like_tau1.weight)
        )

        fake_factor_tau1 = (norm_tau1 * ratio_tau1)

        norm_tau2 = (
            np.sum(df.data.SR_like.weight)
            / np.sum(df.data.AR_like_tau2.weight)
        )

        fake_factor_tau2 = (norm_tau2 * ratio_tau2)

        print(f"tau1 norm = {norm_tau1:.4f}")
        print(f"tau2 norm = {norm_tau2:.4f}")

    else:
        for group_name, ar_mask in group_tau1_masks:

            print_parts = [f"[{group_name}]"]

            # ---------------- Tau 1 ----------------
            if model_tau1 is not None:

                sr_tau1_mask = sr_tau1_masks[group_name]
                ar_tau1_mask = ar_tau1_masks[group_name]

                norm_tau1 = (
                    np.sum(df.data.SR_like.weight[sr_tau1_mask])
                    / np.sum(df.data.AR_like_tau1.weight[ar_tau1_mask])
                )

                fake_factor_tau1[ar_mask] = (
                    norm_tau1 * ratio_tau1[ar_mask]
                )

                print_parts.append(f"tau1 norm = {norm_tau1:.4f}")

        for group_name, ar_mask in group_tau2_masks:
            # ---------------- Tau 2 ----------------
            if model_tau2 is not None:

                sr_tau2_mask = sr_tau2_masks[group_name]
                ar_tau2_mask = ar_tau2_masks[group_name]

                norm_tau2 = (
                    np.sum(df.data.SR_like.weight[sr_tau2_mask])
                    / np.sum(df.data.AR_like_tau2.weight[ar_tau2_mask])
                )

                fake_factor_tau2[ar_mask] = (
                    norm_tau2 * ratio_tau2[ar_mask]
                )

                print_parts.append(f"tau2 norm = {norm_tau2:.4f}")

            print(", ".join(print_parts))

    # ----- number of FF over 3 -----
    print('Tau 1 FF over 3.0')
    for x in fake_factor_tau1:
            if x > 3.0: print(x)
    print('Tau 2 FF over 3.0')
    for x in fake_factor_tau1:
            if x > 3.0: print(x)

    # ------------------------------------------------------------------
    # Optional clipping + output assignment
    # ------------------------------------------------------------------
    if fake_factor_tau1 is not None:
        df.AR_tau1[f"ff_unclipped_dnn_tau1{suffix}"] = fake_factor_tau1
        fake_factor_tau1 = np.clip(fake_factor_tau1, 0, 1)
        df.AR_tau1[f"ff_dnn_tau1{suffix}"] = fake_factor_tau1            
    else:
        print("FF for tau 1 is None")

    if fake_factor_tau2 is not None:
        df.AR_tau2[f"ff_unclipped_dnn_tau2{suffix}"] = fake_factor_tau2
        fake_factor_tau2 = np.clip(fake_factor_tau2, 0, 1)
        df.AR_tau2[f"ff_dnn_tau2{suffix}"] = fake_factor_tau2
    else:
        print("FF for tau 2 is None")


#used
def calculate_fake_factors_incl(
    df,
    incl,
    model: t.nn.Module = None,
    training_variables=None,
    grouping_variable=None,
    grouping_definition=None,
    output_suffix=None,
):

    X_tau = test_data(df.AR, training_variables)
    X_tensor = t.from_numpy(X_tau.X).float()

    eps = 1e-6
    suffix = f"_{output_suffix}" if output_suffix else ""

    if isinstance(grouping_variable, list):
        grouping_var_1 = grouping_variable[0]
        grouping_var_2 = grouping_variable[1]
    elif grouping_variable is not None:
        grouping_var_1 = grouping_variable
        grouping_var_2 = grouping_variable

    # ------------------------------------------------------------------
    # Helper to run inference only if model exists
    # ------------------------------------------------------------------
    def _compute_ratio(model):
        if model is None:
            return None

        X_prepared = _prepare_input_tensor(model, X_tensor, df.AR)

        with t.no_grad():
            f = model(X_prepared).cpu().numpy().flatten()

        f_ = np.clip(f, eps, 1 - eps)

        return f_ / (1.0 - f_)

    ratio = _compute_ratio(model)

    # ------------------------------------------------------------------
    # Allocate outputs only for existing models
    # ------------------------------------------------------------------
    fake_factor = (
        np.zeros_like(ratio) if ratio is not None else None
    )

    if grouping_variable is None:
        ar_group_values = np.asarray(df.AR)
    else:
        ar_tau1_group_values = np.asarray(df.AR_tau1[grouping_var_1])
        ar_tau2_group_values = np.asarray(df.AR_tau2[grouping_var_2])

    
    group_masks = _build_group_masks(
        ar_group_values,
        grouping_definition,
    )

    # ------------------------------------------------------------------
    # Build masks once outside the loop
    # ------------------------------------------------------------------
    
    if model is not None and grouping_definition is not None:
        #Todo!
        sr_tau1_masks = dict(
            _build_group_masks(
                np.asarray(df.data.SR_like[grouping_var_1]),
                grouping_definition,
            )
        )

        ar_tau1_masks = dict(
            _build_group_masks(
                np.asarray(df.data.AR_like_tau1[grouping_var_1]),
                grouping_definition,
            )
        )
    
    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    if grouping_variable is None and model is not None:

        norm = (
            np.sum(df.data.SR_like.weight)
            / np.sum(df.data.AR_like.weight)
        )

        fake_factor = (norm * ratio)

        

        print(f"norm = {norm:.4f}")

    else:
        for group_name, ar_mask in group_tau1_masks:

            print_parts = [f"[{group_name}]"]

            # ---------------- Tau 1 ----------------
            if model_tau1 is not None:

                sr_tau1_mask = sr_tau1_masks[group_name]
                ar_tau1_mask = ar_tau1_masks[group_name]

                norm_tau1 = (
                    np.sum(df.data.SR_like.weight[sr_tau1_mask])
                    / np.sum(df.data.AR_like_tau1.weight[ar_tau1_mask])
                )

                fake_factor_tau1[ar_mask] = (
                    norm_tau1 * ratio_tau1[ar_mask]
                )

                print_parts.append(f"tau1 norm = {norm_tau1:.4f}")

        for group_name, ar_mask in group_tau2_masks:
            # ---------------- Tau 2 ----------------
            if model_tau2 is not None:

                sr_tau2_mask = sr_tau2_masks[group_name]
                ar_tau2_mask = ar_tau2_masks[group_name]

                norm_tau2 = (
                    np.sum(df.data.SR_like.weight[sr_tau2_mask])
                    / np.sum(df.data.AR_like_tau2.weight[ar_tau2_mask])
                )

                fake_factor_tau2[ar_mask] = (
                    norm_tau2 * ratio_tau2[ar_mask]
                )

                print_parts.append(f"tau2 norm = {norm_tau2:.4f}")

            print(", ".join(print_parts))

    # ----- number of FF over 3 -----
    print('FF over 3.0')
    for x in fake_factor:
            if x > 3.0: print(x)
    # ------------------------------------------------------------------
    # Optional clipping + output assignment
    # ------------------------------------------------------------------
    if fake_factor is not None:
        df.AR[f"ff_unclipped_dnn_incl_{incl}{suffix}"] = fake_factor
        fake_factor = np.clip(fake_factor, 0, 1)
        df.AR[f"ff_dnn_incl_{incl}{suffix}"] = fake_factor           
    else:
        print("FF is None")




def calculate_fake_factors_in_DR_wjets(
    df,
    model_wjets: t.nn.Module,
    training_variables,
):
    output_suffix = None
    grouping_variable = 'tau_decaymode_2'
    grouping_tdm = (
        (0,),
        (1,),
        (10,),
        (11,),
    )

    grouping_definition = grouping_tdm

    X = test_data(df.AR_like_wjets, training_variables)

    X_tensor = t.from_numpy(X.X).float()

    X_wjets = _prepare_input_tensor(model_wjets, X_tensor, df.AR_like_wjets)

    with t.no_grad():

        f_wjets = model_wjets(X_wjets).cpu().numpy().flatten()

    eps = 1e-6

    f_wjets = np.clip(f_wjets, eps, 1 - eps)

    ratio_wjets = f_wjets / (1.0 - f_wjets)


    fake_factor_wjets = np.zeros_like(ratio_wjets)



    ar_group_values = np.asarray(df.AR_like_wjets[grouping_variable])

    group_masks = _build_group_masks(
        ar_group_values,
        grouping_definition,
    )


    for group_name, ar_mask in group_masks:



        sr_wjets_mask = _build_group_masks(
            np.asarray(df.data.SR_like_wjets[grouping_variable]),
            grouping_definition,
        )

        ar_wjets_mask = _build_group_masks(
            np.asarray(df.data.AR_like_wjets[grouping_variable]),
            grouping_definition,
        )


        # get corresponding mask
        sr_wjets_mask = dict(sr_wjets_mask)[group_name]
        ar_wjets_mask = dict(ar_wjets_mask)[group_name]

        norm_wjets = (
            np.sum(df.data.SR_like_wjets.weight[sr_wjets_mask])
            / np.sum(df.data.AR_like_wjets.weight[ar_wjets_mask])
        )

        fake_factor_wjets[ar_mask] = (
            norm_wjets * ratio_wjets[ar_mask]
        )

        print(
            f"[{group_name}] "
            f"WJets norm = {norm_wjets:.4f}, "
        )

    # optional clipping
    fake_factor_wjets = np.clip(fake_factor_wjets, 0, 1)



    suffix = f"_{output_suffix}" if output_suffix else ""

    df.AR_like_wjets[f"ff_dnn_wjets{suffix}"] = fake_factor_wjets

def calculate_fake_factors_in_DR(
    df,
    model_tau1: t.nn.Module,
    model_tau2: t.nn.Module,
    training_variables,
    grouping_variable=None,
    grouping_definition=None,
    output_suffix=None,
):

    X_tau1 = test_data(df.AR_like_tau1, training_variables)
    X_tau1_tensor = t.from_numpy(X_tau1.X).float()
    X_tau1 = _prepare_input_tensor(model_tau1, X_tau1_tensor, df.AR_like_tau1)

    X_tau2 = test_data(df.AR_like_tau2, training_variables)
    X_tau2_tensor = t.from_numpy(X_tau2.X).float()
    X_tau2 = _prepare_input_tensor(model_tau2, X_tau2_tensor, df.AR_like_tau2)

    with t.no_grad():
        f_tau1 = model_tau1(X_tau1).cpu().numpy().flatten()
        f_tau2 = model_tau2(X_tau2).cpu().numpy().flatten()

    eps = 1e-6

    suffix = f"_{output_suffix}" if output_suffix else ""
    
    if isinstance(grouping_variable, list):
        grouping_var_1 = grouping_variable[0]
        grouping_var_2 = grouping_variable[1]
    elif grouping_variable is not None:
        grouping_var_1 = grouping_variable
        grouping_var_2 = grouping_variable

    f_tau1 = np.clip(f_tau1, eps, 1 - eps)
    f_tau2 = np.clip(f_tau2, eps, 1 - eps)

    ratio_tau1 = f_tau1 / (1.0 - f_tau1)
    ratio_tau2 = f_tau2 / (1.0 - f_tau2)


    fake_factor_tau1 = np.zeros_like(ratio_tau1)
    fake_factor_tau2 = np.zeros_like(ratio_tau2)

    if grouping_variable is None:
        ar_tau1_group_values = np.asarray(df.AR_like_tau1)
        ar_tau2_group_values = np.asarray(df.AR_like_tau2)
    else:
        ar_tau1_group_values = np.asarray(df.AR_like_tau1[grouping_var_1])
        ar_tau2_group_values = np.asarray(df.AR_like_tau2[grouping_var_2])

    group_tau1_masks = _build_group_masks(
        ar_tau1_group_values,
        grouping_definition,
    )

    group_tau2_masks = _build_group_masks(
        ar_tau2_group_values,
        grouping_definition,
    )

    if group_tau1_masks is None and group_tau2_masks is None:
        norm_tau1 = (
            np.sum(df.data.SR_like.weight)
            / np.sum(df.data.AR_like_tau1.weight)
        )

        fake_factor_tau1 = (norm_tau1 * ratio_tau1)


        norm_tau2 = (
            np.sum(df.data.SR_like.weight)
            / np.sum(df.data.AR_like_tau2.weight)
        )

        fake_factor_tau2 = (norm_tau2 * ratio_tau2)

    else:
        for group_name, ar_mask in group_tau1_masks:        
            if isinstance(grouping_variable, list): grouping_variable = grouping_variable[0]

            sr_tau1_mask = _build_group_masks(
                np.asarray(df.data.SR_like[grouping_variable]),
                grouping_definition,
            )

            ar_tau1_mask = _build_group_masks(
                np.asarray(df.data.AR_like_tau1[grouping_variable]),
                grouping_definition,
            )

            # get corresponding mask
            sr_tau1_mask = dict(sr_tau1_mask)[group_name]
            ar_tau1_mask = dict(ar_tau1_mask)[group_name]

            norm_tau1 = (
                np.sum(df.data.SR_like.weight[sr_tau1_mask])
                / np.sum(df.data.AR_like_tau1.weight[ar_tau1_mask])
            )

            fake_factor_tau1[ar_mask] = (
                norm_tau1 * ratio_tau1[ar_mask]
            )

            print(
                f"[{group_name}] "
                f"Tau1 norm = {norm_tau1:.4f}, "
            )

        for group_name, ar_mask in group_tau2_masks:
            if isinstance(grouping_variable, list): grouping_variable = grouping_variable[1]

            sr_tau2_mask = _build_group_masks(
                np.asarray(df.data.SR_like[grouping_variable]),
                grouping_definition,
            )

            ar_tau2_mask = _build_group_masks(
                np.asarray(df.data.AR_like_tau2[grouping_variable]),
                grouping_definition,
            )

            # get corresponding mask
            sr_tau2_mask = dict(sr_tau2_mask)[group_name]
            ar_tau2_mask = dict(ar_tau2_mask)[group_name]

            norm_tau2 = (
                np.sum(df.data.SR_like.weight[sr_tau2_mask])
                / np.sum(df.data.AR_like_tau2.weight[ar_tau2_mask])
            )

            fake_factor_tau2[ar_mask] = (
                norm_tau2 * ratio_tau2[ar_mask]
            )

            print(
                f"[{group_name}] "
                f"Tau2 norm = {norm_tau2:.4f}, "
            )

    # optional clipping
    fake_factor_tau1 = np.clip(fake_factor_tau1, 0, 1)
    fake_factor_tau2 = np.clip(fake_factor_tau2, 0, 1)

    df.AR_like_tau1[f"ff_DR_dnn_tau1{suffix}"] = fake_factor_tau1
    df.AR_like_tau2[f"ff_DR_dnn_tau2{suffix}"] = fake_factor_tau2

def calculate_fake_factors_in_DR_incl(
    df,
    incl,
    model: t.nn.Module,
    training_variables,
    grouping_variable=None,
    grouping_definition=None,
    output_suffix=None,
):

    suffix = f"_{output_suffix}" if output_suffix else ""

    Xx = test_data(df.AR_like, training_variables)
    X_tensor = t.from_numpy(Xx.X).float()
    Xx = _prepare_input_tensor(model, X_tensor, df.AR_like)

    with t.no_grad():
        f = model(Xx).cpu().numpy().flatten()

    eps = 1e-6

    f = np.clip(f, eps, 1 - eps)

    ratio = f / (1.0 - f)


    fake_factor = np.zeros_like(ratio)

    if grouping_variable is None:
        ar_group_values = np.asarray(df.AR)
    else:
        ar_group_values = np.asarray(df.AR_tau1[grouping_variable])

    group_masks = _build_group_masks(
        ar_group_values,
        grouping_definition,
    )

    if group_masks is None:
        norm = (
            np.sum(df.data.SR_like.weight)
            / np.sum(df.data.AR_like.weight)
        )

        fake_factor = (norm * ratio)

        print(f"Tau incl norm = {norm:.4f}, ")

    else:
        for group_name, ar_mask in group_masks: 
            sr_mask = _build_group_masks(
                np.asarray(df.data.SR_like[grouping_variable]),
                grouping_definition,
            )

            ar_mask = _build_group_masks(
                np.asarray(df.data.AR_like[grouping_variable]),
                grouping_definition,
            )

            # get corresponding mask
            sr_mask = dict(sr_mask)[group_name]
            ar_mask = dict(ar_mask)[group_name]

            norm = (
                np.sum(df.data.SR_like.weight[sr_mask])
                / np.sum(df.data.AR_like.weight[ar_mask])
            )

            fake_factor[ar_mask] = (
                norm * ratio[ar_mask]
            )

            print(
                f"[{group_name}] "
                f"Tau incl norm = {norm:.4f}, "
            )

    # optional clipping
    if fake_factor is not None:
        df.AR_like[f"ff_DR_unclipped_dnn_incl_{incl}{suffix}"] = fake_factor
        fake_factor = np.clip(fake_factor, 0, 1)
        df.AR_like[f"ff_DR_dnn_incl_{incl}{suffix}"] = fake_factor           
    else:
        print("FF is None")


def calculate_fake_factors_in_DR_ttbar(
    df,
    model_ttbar: t.nn.Module,
    training_variables,
    #grouping_variable,
    #grouping_definition,
    #output_suffix=None,
):

    output_suffix = None
    grouping_variable = 'tau_decaymode_2'
    grouping_tdm = (
        (0,),
        (1,),
        (10,),
        (11,),
    )

    grouping_definition = grouping_tdm

    X = test_data(df.AR_like_ttbar, training_variables)

    X_tensor = t.from_numpy(X.X).float()

    X_ttbar = _prepare_input_tensor(model_ttbar, X_tensor, df.AR_like_ttbar)

    with t.no_grad():

        f_ttbar = model_ttbar(X_ttbar).cpu().numpy().flatten()

    eps = 1e-6

    f_ttbar = np.clip(f_ttbar, eps, 1 - eps)

    ratio_ttbar = f_ttbar / (1.0 - f_ttbar)


    fake_factor_ttbar = np.zeros_like(ratio_ttbar)



    ar_group_values = np.asarray(df.AR_like_ttbar[grouping_variable])

    group_masks = _build_group_masks(
        ar_group_values,
        grouping_definition,
    )


    for group_name, ar_mask in group_masks:



        sr_ttbar_mask = _build_group_masks(
            np.asarray(df.data.SR_like_ttbar[grouping_variable]),
            grouping_definition,
        )

        ar_ttbar_mask = _build_group_masks(
            np.asarray(df.data.AR_like_ttbar[grouping_variable]),
            grouping_definition,
        )

        # get corresponding mask
        sr_ttbar_mask = dict(sr_ttbar_mask)[group_name]
        ar_ttbar_mask = dict(ar_ttbar_mask)[group_name]

        norm_ttbar = (
            np.sum(df.data.SR_like_ttbar.weight[sr_ttbar_mask])
            / np.sum(df.data.AR_like_ttbar.weight[ar_ttbar_mask])
        )

        fake_factor_ttbar[ar_mask] = (
            norm_ttbar * ratio_ttbar[ar_mask]
        )

        print(
            f"[{group_name}] "
            f"ttbar norm = {norm_ttbar:.4f}, "
        )

    # optional clipping
    fake_factor_ttbar = np.clip(fake_factor_ttbar, 0, 1)



    suffix = f"_{output_suffix}" if output_suffix else ""

    df.AR_like_ttbar[f"ff_dnn_ttbar{suffix}"] = fake_factor_ttbar


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

#used
def calculate_fake_factor_dnn(
        df1,
        df2,
        grouping,
):
    _df1 = df1.copy()
    _df2 = df2.copy()

    # factor 0.5 here globally since FF only used for QCD, not Wjets and ttbar
    if grouping == 'tau_decaymode':
        _df1['ff_dnn_tau_dm'] = (0.5 * _df1['ff_dnn_tau1_tau_dm'])
        df1['ff_dnn_tau_dm'] = _df1['ff_dnn_tau_dm']
        _df2['ff_dnn_tau_dm'] = (0.5 * _df2['ff_dnn_tau2_tau_dm'])
        df2['ff_dnn_tau_dm'] = _df2['ff_dnn_tau_dm']

    elif grouping == 'njets':
        _df1['ff_dnn_njets'] = (0.5 * _df1['ff_dnn_tau1_njets'])
        df1['ff_dnn_njets'] = _df1['ff_dnn_njets']
        _df2['ff_dnn_njets'] = (0.5 * _df2['ff_dnn_tau2_njets'])
        df2['ff_dnn_njets'] = _df2['ff_dnn_njets']

# -------------- classic fake factor determination -------------

#used
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

    #df['ff_classic'] = _df['corrected_ff']

    #return df1, df2

     

#  -------------- fake factor statistical model uncertainty determination


def _enable_dropout_only(model: t.nn.Module) -> None:
    """
    Keep the model in eval mode, but activate dropout layers only.
    This avoids BatchNorm running-stat updates during MC-dropout inference.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, t.nn.Dropout):
            module.train()


def _build_normalization_vector_for_views(
    target_view,
    sr_view,
    ar_view,
    grouping_variable,
    grouping_definition,
):
    """Build one normalization value per event in the target view."""
    normalization = np.zeros(target_view.n, dtype=np.float32)

    target_group_values = np.asarray(target_view[grouping_variable])
    target_masks = _build_group_masks(target_group_values, grouping_definition)

    sr_masks = dict(_build_group_masks(
        np.asarray(sr_view[grouping_variable]),
        grouping_definition,
    ))
    ar_masks = dict(_build_group_masks(
        np.asarray(ar_view[grouping_variable]),
        grouping_definition,
    ))

    for group_name, target_mask in target_masks:
        numerator = np.sum(sr_view.weight[sr_masks[group_name]])
        denominator = np.sum(ar_view.weight[ar_masks[group_name]])
        normalization[target_mask] = numerator / denominator if denominator > 0 else 0.0

    return normalization


def build_normalization_vector(
    df,
    grouping_variable,
    grouping_definition,
    process='wjets',
):
    """Build per-event normalization factors for df.AR."""
    process_views = {
        'wjets': (df.AR, df.data.SR_like_wjets, df.data.AR_like_wjets),
        'qcd': (df.AR, df.data.SR_like_qcd, df.data.AR_like_qcd),
        'ttbar': (df.AR, df.data.SR_like_ttbar, df.data.AR_like_ttbar),
    }

    if process not in process_views:
        raise ValueError(f"Unknown process '{process}'. Use 'wjets', 'qcd', or 'ttbar'.")

    target_view, sr_view, ar_view = process_views[process]
    return _build_normalization_vector_for_views(
        target_view,
        sr_view,
        ar_view,
        grouping_variable,
        grouping_definition,
    )


def build_normalization_vector_in_DR(
    df,
    process,
    grouping_variable,
    grouping_definition,
):
    """Build per-event normalization factors for df.AR_like_process."""
    target_view = getattr(df, f'AR_like_{process}')
    sr_view = getattr(df.data, f'SR_like_{process}')
    ar_view = getattr(df.data, f'AR_like_{process}')

    return _build_normalization_vector_for_views(
        target_view,
        sr_view,
        ar_view,
        grouping_variable,
        grouping_definition,
    )


def _calculate_fake_factor_mean_std_for_view_per_model(
    df_view,
    models,
    training_variables,
    normalization,
    device: t.device | None = None,
):
    """
    Compute FF mean/std by processing all events per model (no batching).
    """
    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    print(f'[INFO] Using device: {device}')
    print('[INFO] Building full input tensor...')

    X = test_data(df_view, training_variables)
    X_tensor = t.from_numpy(X.X).float()
    X = _prepare_input_tensor(models[0], X_tensor, df_view).to(device)

    normalization = t.from_numpy(normalization).float().to(device)

    n_events = df_view.n
    n_models = len(models)

    print(f'[INFO] Events: {n_events:,}')
    print(f'[INFO] Models: {n_models}')
    print('[INFO] Inference mode: full events per model')

    sum_ff = t.zeros(n_events, dtype=t.float32, device=device)
    sum_sq_ff = t.zeros(n_events, dtype=t.float32, device=device)

    start_time = time.time()

    with t.inference_mode():
        for i, model in enumerate(models, start=1):
            model.to(device)
            model.eval()

            f = model(X).squeeze()
            f = t.clamp(f, 1e-6, 1 - 1e-6)
            ratio = f / (1.0 - f)
            ff = ratio * normalization
            ff = t.clamp(ff, 0, 1)

            sum_ff += ff
            sum_sq_ff += ff * ff

            model.cpu()  # free GPU memory after each model

            elapsed = time.time() - start_time
            speed = i / elapsed if elapsed > 0 else 0
            remaining = n_models - i
            eta = remaining / speed if speed > 0 else 0

            print(
                f'\r[PROGRESS] Model {i}/{n_models} | '
                f'{100.0 * i / n_models:6.2f}% | '
                f'{speed:,.2f} models/s | '
                f'ETA {eta/60:.2f} min',
                end='',
                flush=True,
            )

    mean_ff = sum_ff / n_models
    var_ff = (sum_sq_ff / n_models) - mean_ff * mean_ff
    var_ff = t.clamp(var_ff, min=0)
    std_ff = t.sqrt(var_ff)

    print('\n[INFO] Inference complete.')

    return mean_ff.cpu().numpy(), std_ff.cpu().numpy()


def _calculate_fake_factor_mean_std_for_view_per_model_per_mask(
    df_view,
    model,
    training_variables,
    normalization,
    device: t.device | None = None,
):
    """
    Compute FF mean/std by processing all events per model (no batching).
    """
    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    print(f'[INFO] Using device: {device}')
    print('[INFO] Building full input tensor...')

    X = test_data(df_view, training_variables)
    X_tensor = t.from_numpy(X.X).float()
    X = _prepare_input_tensor(model, X_tensor, df_view).to(device)

    normalization = t.from_numpy(normalization).float().to(device)

    n_events = df_view.n
    n_masks = 100

    print(f'[INFO] Events: {n_events:,}')
    print(f'[INFO] Models: {n_masks}')
    print('[INFO] Inference mode: full events per model')

    sum_ff = t.zeros(n_events, dtype=t.float32, device=device)
    sum_sq_ff = t.zeros(n_events, dtype=t.float32, device=device)

    start_time = time.time()

    with t.inference_mode():
        for i in range(1, n_masks + 1):
            model.to(device)
            _enable_dropout_only(model)

            f = model(X).squeeze()
            f = t.clamp(f, 1e-6, 1 - 1e-6)
            ratio = f / (1.0 - f)
            ff = ratio * normalization
            ff = t.clamp(ff, 0, 1)

            sum_ff += ff
            sum_sq_ff += ff * ff

            model.cpu()  # free GPU memory after each model

            elapsed = time.time() - start_time
            speed = i / elapsed if elapsed > 0 else 0
            remaining = n_masks - i
            eta = remaining / speed if speed > 0 else 0

            print(
                f'\r[PROGRESS] Model {i}/{n_masks} | '
                f'{100.0 * i / n_masks:6.2f}% | '
                f'{speed:,.2f} models/s | '
                f'ETA {eta/60:.2f} min',
                end='',
                flush=True,
            )

    mean_ff = sum_ff / n_masks
    var_ff = (sum_sq_ff / n_masks) - mean_ff * mean_ff
    var_ff = t.clamp(var_ff, min=0)
    std_ff = t.sqrt(var_ff)

    print('\n[INFO] Inference complete.')

    return mean_ff.cpu().numpy(), std_ff.cpu().numpy()


def calculate_fake_factor_mean_std(
    df,
    models,
    training_variables,
    grouping_variable,
    grouping_definition,
    process='wjets',
    output_mean='fake_factor_mean',
    output_std='fake_factor_std',
    device: t.device | None = None,
):
    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    print('[INFO] Computing normalization...')
    normalization = build_normalization_vector(
        df,
        grouping_variable,
        grouping_definition,
        process,
    )

    mean_result, std_result = _calculate_fake_factor_mean_std_for_view_per_model(
        df.AR,
        models,
        training_variables,
        normalization,
        device=device,
    )

    df.AR[output_mean] = mean_result
    df.AR[output_std] = std_result
    return df


def calculate_fake_factor_mean_std_dropout_mask_variation(
    df,
    model,
    training_variables,
    grouping_variable,
    grouping_definition,
    process='wjets',
    output_mean='fake_factor_mean',
    output_std='fake_factor_std',
    device: t.device | None = None,
):
    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    print('[INFO] Computing normalization...')
    normalization = build_normalization_vector(
        df,
        grouping_variable,
        grouping_definition,
        process,
    )

    mean_result, std_result = _calculate_fake_factor_mean_std_for_view_per_model_per_mask(
        df.AR,
        model,
        training_variables,
        normalization,
        device=device,
    )

    df.AR[output_mean] = mean_result
    df.AR[output_std] = std_result
    return df


def calculate_fake_factor_mean_std_in_DR(
    df,
    models,
    training_variables,
    grouping_variable,
    grouping_definition,
    process='wjets',
    output_mean='fake_factor_mean',
    output_std='fake_factor_std',
    device: t.device | None = None,
):
    if process not in {'wjets', 'qcd', 'ttbar'}:
        raise ValueError("calculate_fake_factor_mean_std_batched_in_DR only supports process='wjets', 'qcd', or 'ttbar'.")

    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    print('[INFO] Computing normalization...')
    normalization = build_normalization_vector_in_DR(
        df,
        process,
        grouping_variable,
        grouping_definition,
    )

    target_view = getattr(df, f'AR_like_{process}')

    mean_result, std_result = _calculate_fake_factor_mean_std_for_view_per_model(
        target_view,
        models,
        training_variables,
        normalization,
        device=device,
    )

    target_view[output_mean] = mean_result
    target_view[output_std] = std_result
    return df


def calculate_fake_factor_mean_std_in_DR_dropout_mask_variation(

    df,
    model,
    training_variables,
    grouping_variable,
    grouping_definition,
    process='wjets',
    output_mean='fake_factor_mean',
    output_std='fake_factor_std',
    device: t.device | None = None,
):
    """
    returns fake factor mean and std for 100 different model masks
    """
    if process not in {'wjets', 'qcd', 'ttbar'}:
        raise ValueError("calculate_fake_factor_mean_std_batched_in_DR only supports process='wjets', 'qcd', or 'ttbar'.")

    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    print('[INFO] Computing normalization...')
    normalization = build_normalization_vector_in_DR(
        df,
        process,
        grouping_variable,
        grouping_definition,
    )

    target_view = getattr(df, f'AR_like_{process}')

    mean_result, std_result = _calculate_fake_factor_mean_std_for_view_per_model_per_mask(
        target_view,
        model,
        training_variables,
        normalization,
        device=device,
    )

    target_view[output_mean] = mean_result
    target_view[output_std] = std_result
    return df



# -----------------------