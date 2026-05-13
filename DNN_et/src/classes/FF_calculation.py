import torch as t
import correctionlib as cr
import numpy as np
import pandas as pd
from .NeuralNetworks import load_model, FoldCombinedDNN
from .DataHandling import test_data



def _prepare_input_tensor(model: t.nn.Module, X_tensor: t.Tensor, df_ar) -> t.Tensor:
    """Return the correctly shaped input tensor for the given model type."""
    if isinstance(model, FoldCombinedDNN):
        event_ids = t.from_numpy(np.asarray(df_ar['event']%2, dtype=np.float32))
        return t.cat([event_ids.unsqueeze(0), X_tensor.T], dim=0)  # [1 + n_features, N]
    return X_tensor  # [N, n_features]


def _build_group_masks(values, grouping_definition):

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



def calculate_fake_factors(
    df,
    model_wjets: t.nn.Module,
    model_qcd: t.nn.Module,
    training_variables,
    grouping_variable,
    grouping_definition,
    output_suffix=None,
):


    X = test_data(df.AR, training_variables)

    X_tensor = t.from_numpy(X.X).float()

    X_wjets = _prepare_input_tensor(model_wjets, X_tensor, df.AR)
    X_qcd = _prepare_input_tensor(model_qcd, X_tensor, df.AR)

    with t.no_grad():

        f_wjets = model_wjets(X_wjets).cpu().numpy().flatten()
        f_qcd = model_qcd(X_qcd).cpu().numpy().flatten()

    eps = 1e-6

    f_wjets = np.clip(f_wjets, eps, 1 - eps)
    f_qcd = np.clip(f_qcd, eps, 1 - eps)

    ratio_wjets = f_wjets / (1.0 - f_wjets)
    ratio_qcd = f_qcd / (1.0 - f_qcd)


    fake_factor_wjets = np.zeros_like(ratio_wjets)
    fake_factor_qcd = np.zeros_like(ratio_qcd)



    ar_group_values = np.asarray(df.AR[grouping_variable])

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

        sr_qcd_mask = _build_group_masks(
            np.asarray(df.data.SR_like_qcd[grouping_variable]),
            grouping_definition,
        )

        ar_qcd_mask = _build_group_masks(
            np.asarray(df.data.AR_like_qcd[grouping_variable]),
            grouping_definition,
        )

        # get corresponding mask
        sr_wjets_mask = dict(sr_wjets_mask)[group_name]
        ar_wjets_mask = dict(ar_wjets_mask)[group_name]

        sr_qcd_mask = dict(sr_qcd_mask)[group_name]
        ar_qcd_mask = dict(ar_qcd_mask)[group_name]



        norm_wjets = (
            np.sum(df.data.SR_like_wjets.weight[sr_wjets_mask])
            / np.sum(df.data.AR_like_wjets.weight[ar_wjets_mask])
        )

        norm_qcd = (
            np.sum(df.data.SR_like_qcd.weight[sr_qcd_mask])
            / np.sum(df.data.AR_like_qcd.weight[ar_qcd_mask])
        )


        fake_factor_wjets[ar_mask] = (
            norm_wjets * ratio_wjets[ar_mask]
        )

        fake_factor_qcd[ar_mask] = (
            norm_qcd * ratio_qcd[ar_mask]
        )

        print(
            f"[{group_name}] "
            f"WJets norm = {norm_wjets:.4f}, "
            f"QCD norm = {norm_qcd:.4f}"
        )

    # optional clipping
    fake_factor_wjets = np.clip(fake_factor_wjets, 0, 1)
    fake_factor_qcd = np.clip(fake_factor_qcd, 0, 1)



    suffix = f"_{output_suffix}" if output_suffix else ""

    df.AR[f"ff_dnn_wjets{suffix}"] = fake_factor_wjets
    df.AR[f"ff_dnn_qcd{suffix}"] = fake_factor_qcd

def calculate_fake_factors_in_DR_wjets(
    df,
    model_wjets: t.nn.Module,
    training_variables,
    grouping_variable,
    grouping_definition,
    output_suffix=None,
):


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

        sr_qcd_mask = _build_group_masks(
            np.asarray(df.data.SR_like_qcd[grouping_variable]),
            grouping_definition,
        )

        ar_qcd_mask = _build_group_masks(
            np.asarray(df.data.AR_like_qcd[grouping_variable]),
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

def calculate_fake_factors_in_DR_qcd(
    df,
    model_qcd: t.nn.Module,
    training_variables,
    grouping_variable,
    grouping_definition,
    output_suffix=None,
):


    X = test_data(df.AR_like_qcd, training_variables)

    X_tensor = t.from_numpy(X.X).float()

    X_qcd = _prepare_input_tensor(model_qcd, X_tensor, df.AR_like_qcd)

    with t.no_grad():

        f_qcd = model_qcd(X_qcd).cpu().numpy().flatten()

    eps = 1e-6

    f_qcd = np.clip(f_qcd, eps, 1 - eps)

    ratio_qcd = f_qcd / (1.0 - f_qcd)


    fake_factor_qcd = np.zeros_like(ratio_qcd)



    ar_group_values = np.asarray(df.AR_like_qcd[grouping_variable])

    group_masks = _build_group_masks(
        ar_group_values,
        grouping_definition,
    )


    for group_name, ar_mask in group_masks:



        sr_qcd_mask = _build_group_masks(
            np.asarray(df.data.SR_like_qcd[grouping_variable]),
            grouping_definition,
        )

        ar_qcd_mask = _build_group_masks(
            np.asarray(df.data.AR_like_qcd[grouping_variable]),
            grouping_definition,
        )

        # get corresponding mask
        sr_qcd_mask = dict(sr_qcd_mask)[group_name]
        ar_qcd_mask = dict(ar_qcd_mask)[group_name]

        norm_qcd = (
            np.sum(df.data.SR_like_qcd.weight[sr_qcd_mask])
            / np.sum(df.data.AR_like_qcd.weight[ar_qcd_mask])
        )

        fake_factor_qcd[ar_mask] = (
            norm_qcd * ratio_qcd[ar_mask]
        )

        print(
            f"[{group_name}] "
            f"QCD norm = {norm_qcd:.4f}, "
        )

    # optional clipping
    fake_factor_qcd = np.clip(fake_factor_qcd, 0, 1)



    suffix = f"_{output_suffix}" if output_suffix else ""

    df.AR_like_qcd[f"ff_dnn_qcd{suffix}"] = fake_factor_qcd



def evaluate_compound_ff_correction(correction_set, compound_name: str, df: pd.DataFrame) -> np.ndarray:
    compound_correction = correction_set.compound[compound_name]
    expected_inputs = [input_spec.name for input_spec in compound_correction.inputs]

    input_values = {
        'tau_decaymode_2': df.tau_decaymode_2,
        'eta_1': df.eta_1,
        'eta_2': df.eta_2,
        'jeta_1': df.jeta_1,
        'jeta_2': df.jeta_2,
        'jpt_1': df.jpt_1,
        'jpt_2': df.jpt_2,
        'met': df.met,
        'deltaR_ditaupair': df.deltaR_ditaupair,
        'deltaR_1j1': df.deltaR_1j1,
        'deltaR_12j1': df.deltaR_12j1,
        'pt_ttjj': df.pt_ttjj,
        'mass_2': df.mass_2,
        'mt_tot': df.mt_tot,
        'm_vis': df.m_vis,
        'iso_1': df.iso_1,
        'njets': df.njets,
        'syst': 'nominal',
    }

    missing_inputs = [name for name in expected_inputs if name not in input_values]
    if missing_inputs:
        raise KeyError(f'Missing input mapping for correction {compound_name}: {missing_inputs}')

    ordered_inputs = [input_values[name] for name in expected_inputs]
    return compound_correction.evaluate(*ordered_inputs)


def calculate_fake_factor_classic(
        df,
        ):
    _df = df.copy()
    ff = cr.CorrectionSet.from_file('/work/mmoser/TauFakeFactors/workdir/ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz')

    frac = ff['process_fractions']


    ff_wjets = ff['Wjets_fake_factors']
    ff_qcd = ff['QCD_fake_factors']
    ff_ttbar = ff['ttbar_fake_factors']

    corr = cr.CorrectionSet.from_file('/work/mmoser/TauFakeFactors/workdir/ff_2026_01_19_check_variable/2018/FF_corrections_et.json.gz')


    _df["wjets_classic_ff"] = ff_wjets.evaluate(
        _df.pt_2.values,
        _df.njets.values,
        _df.pt_1.values,
        "nominal",
    )



    _df['qcd_classic_ff'] = ff_qcd.evaluate(
        _df.pt_2.values,
        _df.njets.values,
        "nominal",
    )

    _df['ttbar_classic_ff'] = ff_ttbar.evaluate(
        _df.pt_2.values,
        _df.njets.values,
        "nominal",
    )

    _df["wjets_corrected_classic_ff"] = _df["wjets_classic_ff"] * evaluate_compound_ff_correction(
        corr,
        "Wjets_compound_correction",
        _df,
    ) * corr["Wjets_DR_SR_correction"].evaluate(
        _df.pt_tt,
        _df.njets,
        "nominal",
    )

    _df["qcd_corrected_classic_ff"] = _df["qcd_classic_ff"] * evaluate_compound_ff_correction(
        corr,
        "QCD_compound_correction",
        _df,
    ) * corr["QCD_DR_SR_correction"].evaluate(
        _df.pt_tt,
        _df.njets,
        "nominal",
    )

    _df["ttbar_corrected_classic_ff"] = _df["ttbar_classic_ff"] * evaluate_compound_ff_correction(
        corr,
        "ttbar_compound_correction",
        _df,
    )

    _df['process_fraction_wjets'] = frac.evaluate(
        'Wjets',
        _df.mt_1.values,
        _df.njets.values,
        'nominal'
    )

    _df['process_fraction_qcd'] = frac.evaluate(
        'QCD',
        _df.mt_1.values,
        _df.njets.values,
        'nominal'
    )

    _df['process_fraction_ttbar'] = frac.evaluate(
        'ttbar',
        _df.mt_1.values,
        _df.njets.values,
        'nominal'
    )

    _df['corrected_ff'] = _df['process_fraction_wjets'] * _df['wjets_corrected_classic_ff'] + _df['process_fraction_qcd'] * _df['qcd_corrected_classic_ff'] + _df['process_fraction_ttbar'] * _df['ttbar_corrected_classic_ff']

    df['ff_classic'] = _df['corrected_ff']

    return df


def calculate_fake_factor_dnn(
        df,
        grouping,
):
    _df = df.copy()

    _ff_file = cr.CorrectionSet.from_file('/work/mmoser/TauFakeFactors/workdir/ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz')
    _corr_file = cr.CorrectionSet.from_file('/work/mmoser/TauFakeFactors/workdir/ff_2026_01_19_check_variable/2018/FF_corrections_et.json.gz')

    _frac = _ff_file['process_fractions']
    _ff_ttbar = _ff_file['ttbar_fake_factors']

    _df['ttbar_classic_ff'] = _ff_ttbar.evaluate(
        _df.pt_2.values,
        _df.njets.values,
        "nominal",
    )
    _df['ttbar_corrected_classic_ff'] = _df['ttbar_classic_ff'] * evaluate_compound_ff_correction(
        _corr_file,
        'ttbar_compound_correction',
        _df,
    )

    _df['process_fraction_wjets'] = _frac.evaluate('Wjets', _df.mt_1.values, _df.njets.values, 'nominal')
    _df['process_fraction_qcd'] = _frac.evaluate('QCD', _df.mt_1.values, _df.njets.values, 'nominal')
    _df['process_fraction_ttbar'] = _frac.evaluate('ttbar', _df.mt_1.values, _df.njets.values, 'nominal')


    if grouping == 'tau_decaymode':
        _df['ff_dnn_tdm'] = (
            _df['process_fraction_wjets'] * _df['ff_dnn_wjets_tdm']
            + _df['process_fraction_qcd'] * _df['ff_dnn_qcd_tdm']
            + _df['process_fraction_ttbar'] * _df['ttbar_corrected_classic_ff']
        )
        df['ff_dnn_tdm'] = _df['ff_dnn_tdm']

    elif grouping == 'njets':
        _df['ff_dnn_njets'] = (
            _df['process_fraction_wjets'] * _df['ff_dnn_wjets_njets']
            + _df['process_fraction_qcd'] * _df['ff_dnn_qcd_njets']
            + _df['process_fraction_ttbar'] * _df['ttbar_corrected_classic_ff']
        )
        df['ff_dnn_njets'] = _df['ff_dnn_njets']
        

    

    

