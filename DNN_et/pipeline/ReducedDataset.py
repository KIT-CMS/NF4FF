import numpy as np
from classes import load_fold_combined_model, load_data, load_variables
from pathlib import Path
import torch as t
import pandas as pd
import logging
from classes import FeatureRegistry, FeatureStore


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / 'Enrichment_models'
GROUPING_NAMES = ['tau_decaymode_2', 'njets']

DATA_PATH = PROJECT_ROOT / 'data' / 'dataframe_complete.feather'
MASKS_PATH = PROJECT_ROOT / 'configs' / 'masks.yaml'
VARIABLES_ENRICHMENT_PATH = PROJECT_ROOT / 'configs' / 'training_variables_enrichment.yaml'
NBINS = 40

FEATURE_STORE_DIR_WJETS = PROJECT_ROOT / 'data' / 'features' / 'reduced_dataset' / 'wjets'
FEATURE_REGISTRY_PATH_WJETS = FEATURE_STORE_DIR_WJETS / 'feature_registry.json'
FEATURE_STORE_DIR_QCD = PROJECT_ROOT / 'data' / 'features' / 'reduced_dataset' / 'qcd'
FEATURE_REGISTRY_PATH_QCD = FEATURE_STORE_DIR_QCD / 'feature_registry.json'
QCD_WEIGHT_STORE_DIR_WJETS = PROJECT_ROOT / 'data' / 'features' / 'wjets'
QCD_WEIGHT_STORE_DIR_QCD = PROJECT_ROOT / 'data' / 'features' / 'qcd'
DEFAULT_FEATURE_REGISTRY_PATH = PROJECT_ROOT / 'data' / 'features' / 'feature_registry.json'

PROCESSES = ['diboson', 'DYjets', 'ST', 'ttbar', 'embedding']
PROCESSES_QCD = ['wjets', 'diboson', 'DYjets', 'ST', 'ttbar', 'embedding']


def _configure_output_root(output_root):
    global CHECKPOINT_DIR
    global DATA_PATH
    global FEATURE_STORE_DIR_WJETS
    global FEATURE_REGISTRY_PATH_WJETS
    global FEATURE_STORE_DIR_QCD
    global FEATURE_REGISTRY_PATH_QCD
    global QCD_WEIGHT_STORE_DIR_WJETS
    global QCD_WEIGHT_STORE_DIR_QCD
    global DEFAULT_FEATURE_REGISTRY_PATH

    output_root = Path(output_root)
    CHECKPOINT_DIR = output_root / 'Enrichment_models'
    DATA_PATH = output_root / 'data' / 'dataframe_complete.feather'
    feature_root = output_root / 'data' / 'features'
    FEATURE_STORE_DIR_WJETS = feature_root / 'reduced_dataset' / 'wjets'
    FEATURE_REGISTRY_PATH_WJETS = FEATURE_STORE_DIR_WJETS / 'feature_registry.json'
    FEATURE_STORE_DIR_QCD = feature_root / 'reduced_dataset' / 'qcd'
    FEATURE_REGISTRY_PATH_QCD = FEATURE_STORE_DIR_QCD / 'feature_registry.json'
    QCD_WEIGHT_STORE_DIR_WJETS = feature_root / 'wjets'
    QCD_WEIGHT_STORE_DIR_QCD = feature_root / 'qcd'
    DEFAULT_FEATURE_REGISTRY_PATH = feature_root / 'feature_registry.json'


def _reduced_weight_name(process, grouping):
    return f"reduced_weight_{process}_{grouping}_nominal"


def _nn_output_name(process, grouping):
    return f"nn_output_{process}_{grouping}"


def _register_reduced_features(feature_names, feature_store_path, legacy_name):
    default_registry = FeatureRegistry(DEFAULT_FEATURE_REGISTRY_PATH)
    default_registry.index.pop(legacy_name, None)
    default_registry.register(feature_names, feature_store_path)
    default_registry.save()


def _load_group_model(process, grouping):
    checkpoint_dir = CHECKPOINT_DIR / process / grouping
    even_model_path = checkpoint_dir / 'fold_even'
    odd_model_path = checkpoint_dir / 'fold_odd'

    missing = [
        path for path in (even_model_path, odd_model_path)
        if not path.is_dir()
    ]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        task_name = {
            "wjets": "TrainEnrichmentWjetsV2",
            "qcd": "TrainEnrichmentQCDV2",
        }[process]
        raise FileNotFoundError(
            f"Missing enrichment model directories: {missing_paths}. "
            f"Run {task_name} first."
        )

    return load_fold_combined_model(
        even_model_path=even_model_path,
        odd_model_path=odd_model_path,
    )



def equi_populated_bins(data, n_bins):
    data = np.asarray(data)
    finite_data = data[np.isfinite(data)]
    if finite_data.size == 0:
        raise ValueError("Cannot build bins without finite values.")

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(finite_data, quantiles)

    # The subtraction helper operates in the NN-output dtype. Cast here so
    # np.histogram and the per-event bin assignment use exactly the same edges.
    if np.issubdtype(data.dtype, np.floating):
        bin_edges = bin_edges.astype(data.dtype, copy=False)

    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        raise ValueError("Cannot build bins from a constant NN output.")

    return bin_edges


def sum_process_histograms(
    df,
    bins,
    processes=PROCESSES,
    value_column='nn_output',
    weight_column='weight',
    region_name=None,
):
    hist_sum = np.zeros(len(bins) - 1, dtype=np.float64)

    for process in processes:
        process_view = getattr(df, process)
        if region_name is not None:
            process_view = getattr(process_view, region_name)
        process_df = process_view.events
        values = process_df[value_column].to_numpy(dtype=bins.dtype)
        weights = (
            process_df[weight_column].to_numpy(dtype=np.float64)
            if weight_column is not None
            else None
        )
        hist_proc, _ = np.histogram(values, bins=bins, weights=weights)
        hist_sum += hist_proc

    return hist_sum


def _calculate_reduced_weights(values, bins, subtraction_histogram):
    values = np.asarray(values)
    bins = np.asarray(bins, dtype=values.dtype)
    subtraction_histogram = np.asarray(
        subtraction_histogram,
        dtype=np.float64,
    )

    data_histogram, _ = np.histogram(values, bins=bins)
    if len(subtraction_histogram) != len(data_histogram):
        raise ValueError(
            "Subtraction histogram does not match the number of bins."
        )

    scale_factors = np.ones(len(data_histogram), dtype=np.float64)
    populated = data_histogram != 0
    scale_factors[populated] -= (
        subtraction_histogram[populated] / data_histogram[populated]
    )

    empty_with_subtraction = (~populated) & (subtraction_histogram != 0)
    if empty_with_subtraction.any():
        bad_bins = np.flatnonzero(empty_with_subtraction).tolist()
        raise RuntimeError(
            "Cannot subtract from empty data bins: "
            f"{bad_bins}"
        )

    event_bins = np.searchsorted(bins, values, side="right") - 1
    event_bins[values == bins[-1]] = len(bins) - 2
    in_range = (values >= bins[0]) & (values <= bins[-1])

    reduced_weights = np.ones(values.shape, dtype=np.float64)
    reduced_weights[in_range] = scale_factors[event_bins[in_range]]
    return reduced_weights


def _validate_reduced_subtraction(
    values,
    reduced_weights,
    bins,
    data_histogram,
    subtraction_histogram,
    label,
    diagnostics_path,
):
    reduced_histogram, _ = np.histogram(
        values,
        bins=bins,
        weights=reduced_weights,
    )
    expected_histogram = data_histogram - subtraction_histogram

    negative_bins = expected_histogram < 0
    diagnostics = pd.DataFrame({
        "bin_low": bins[:-1],
        "bin_high": bins[1:],
        "data_yield": data_histogram,
        "subtraction_yield": subtraction_histogram,
        "expected_reduced_yield": expected_histogram,
        "actual_reduced_yield": reduced_histogram,
        "difference": reduced_histogram - expected_histogram,
        "is_negative": negative_bins,
    })
    diagnostics_path = Path(diagnostics_path)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(diagnostics_path, index=False)

    if not np.allclose(
        reduced_histogram,
        expected_histogram,
        rtol=1e-5,
        atol=1e-5,
    ):
        max_difference = np.max(
            np.abs(reduced_histogram - expected_histogram)
        )
        raise RuntimeError(
            f"{label}: reduced-weight closure failed; "
            f"maximum bin difference is {max_difference:.6g}"
        )

    negative_events = reduced_weights < 0
    logger.info(
        "%s: subtraction closure passed; negative bins=%d/%d, "
        "negative event weights=%d/%d",
        label,
        int(negative_bins.sum()),
        len(expected_histogram),
        int(negative_events.sum()),
        len(reduced_weights),
    )


def _predict_fold_output(model, region_df, training_variables, device):
    features = t.as_tensor(
        region_df[training_variables].to_numpy(dtype=np.float32),
        device=device,
    )
    parity = t.as_tensor(
        (region_df["event"] % 2).to_numpy(dtype=np.float32),
        device=device,
    ).unsqueeze(0)
    model_input = t.cat([parity, features.T], dim=0)
    return model(model_input).detach().cpu().numpy()


def reduced_data_wjets(output_root=PROJECT_ROOT):
    _configure_output_root(output_root)
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    df = load_data(DATA_PATH, MASKS_PATH)
    training_variables = load_variables(VARIABLES_ENRICHMENT_PATH)

    for grouping in GROUPING_NAMES:
        model = _load_group_model("wjets", grouping)
        df.load_feature_file(
            QCD_WEIGHT_STORE_DIR_WJETS / f"qcd_weights_{grouping}.feather"
        )

        inference_view = df.full.DR_wjets_without_signs
        inference_df = inference_view.events.copy()

        nn_output_name = _nn_output_name("wjets", grouping)
        inference_df[nn_output_name] = _predict_fold_output(
            model=model,
            region_df=inference_df,
            training_variables=training_variables,
            device=device,
        )

        inference_view[nn_output_name] = inference_df[nn_output_name].to_numpy()

        data_region = df["data"].DR_wjets
        qcd_region = df["data"].DR_wjets_without_signs
        data_values = data_region[nn_output_name].to_numpy(dtype=np.float32)
        bins = equi_populated_bins(data_values, NBINS)

        hist_data, bin_edges = np.histogram(data_values, bins=bins)
        qcd_values = qcd_region[nn_output_name].to_numpy(dtype=np.float32)
        qcd_weights = qcd_region[f"qcd_weight_{grouping}"].to_numpy(dtype=np.float64)
        valid_qcd = np.isfinite(qcd_values) & np.isfinite(qcd_weights)
        hist_qcd, _ = np.histogram(
            qcd_values[valid_qcd],
            bins=bin_edges,
            weights=qcd_weights[valid_qcd],
        )
        hist_processes = sum_process_histograms(
            df,
            bins=bin_edges,
            processes=PROCESSES,
            value_column=nn_output_name,
            region_name="DR_wjets",
        )

        reduced_weights = _calculate_reduced_weights(
            values=data_values,
            bins=bin_edges,
            subtraction_histogram=hist_qcd + hist_processes,
        )
        _validate_reduced_subtraction(
            values=data_values,
            reduced_weights=reduced_weights,
            bins=bin_edges,
            data_histogram=hist_data,
            subtraction_histogram=hist_qcd + hist_processes,
            label=f"W+jets/{grouping}",
            diagnostics_path=(
                FEATURE_STORE_DIR_WJETS
                / f"subtraction_diagnostics_{grouping}.csv"
            ),
        )

        feature_name = _reduced_weight_name("wjets", grouping)
        legacy_name = f"reduced_weight_{grouping}_nominal"
        data_region[feature_name] = reduced_weights

        feature_store_path = Path(FEATURE_STORE_DIR_WJETS) / f'reduced_weight_{grouping}.feather'

        registry = FeatureRegistry(FEATURE_REGISTRY_PATH_WJETS)
        store = FeatureStore(feature_store_path, registry)
        feature_df = pd.DataFrame({
            "row_index": inference_view.events.index,
            "event": inference_view["event"],
            nn_output_name: inference_view[nn_output_name],
        })
        reduced_by_row = pd.Series(
            data_region[feature_name].to_numpy(),
            index=data_region.events.index,
        )
        feature_df[feature_name] = feature_df["row_index"].map(reduced_by_row)
        store.write(feature_df)
        store.save()
        registry.index.pop(legacy_name, None)
        registry.save()

        _register_reduced_features(
            [nn_output_name, feature_name],
            feature_store_path,
            legacy_name,
        )

    return df




def reduced_data_qcd(output_root=PROJECT_ROOT):
    _configure_output_root(output_root)
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    df = load_data(DATA_PATH, MASKS_PATH)
    training_variables = load_variables(VARIABLES_ENRICHMENT_PATH)

    for grouping in GROUPING_NAMES:
        model = _load_group_model("qcd", grouping)

        inference_view = df.full.DR_qcd
        inference_df = inference_view.events.copy()

        nn_output_name = _nn_output_name("qcd", grouping)
        inference_df[nn_output_name] = _predict_fold_output(
            model=model,
            region_df=inference_df,
            training_variables=training_variables,
            device=device,
        )

        inference_view[nn_output_name] = inference_df[nn_output_name].to_numpy()

        data_region = df["data"].DR_qcd
        data_values = data_region[nn_output_name].to_numpy(dtype=np.float32)
        bins = equi_populated_bins(data_values, NBINS)

        hist_data, bin_edges = np.histogram(data_values, bins=bins)

        hist_processes = sum_process_histograms(
            df,
            bins=bin_edges,
            processes=PROCESSES_QCD,
            value_column=nn_output_name,
            region_name="DR_qcd",
        )

        reduced_weights = _calculate_reduced_weights(
            values=data_values,
            bins=bin_edges,
            subtraction_histogram=hist_processes,
        )
        _validate_reduced_subtraction(
            values=data_values,
            reduced_weights=reduced_weights,
            bins=bin_edges,
            data_histogram=hist_data,
            subtraction_histogram=hist_processes,
            label=f"QCD/{grouping}",
            diagnostics_path=(
                FEATURE_STORE_DIR_QCD
                / f"subtraction_diagnostics_{grouping}.csv"
            ),
        )

        feature_name = _reduced_weight_name("qcd", grouping)
        legacy_name = f"reduced_weight_{grouping}_nominal"
        data_region[feature_name] = reduced_weights

        feature_store_path = Path(FEATURE_STORE_DIR_QCD) / f'reduced_weight_{grouping}.feather'

        registry = FeatureRegistry(FEATURE_REGISTRY_PATH_QCD)
        store = FeatureStore(feature_store_path, registry)
        feature_df = pd.DataFrame({
            "row_index": inference_view.events.index,
            "event": inference_view["event"],
            nn_output_name: inference_view[nn_output_name],
        })
        reduced_by_row = pd.Series(
            data_region[feature_name].to_numpy(),
            index=data_region.events.index,
        )
        feature_df[feature_name] = feature_df["row_index"].map(reduced_by_row)
        store.write(feature_df)
        store.save()
        registry.index.pop(legacy_name, None)
        registry.save()

        _register_reduced_features(
            [nn_output_name, feature_name],
            feature_store_path,
            legacy_name,
        )

    return df






        
