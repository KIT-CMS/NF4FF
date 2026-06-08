import numpy as np
from classes import load_fold_combined_model, load_data, load_variables
from pathlib import Path
from enrichment import get_my_data
from classes.enrichment_classifier import _calculate_scaled_event_weights_generalized
import torch as t
import pandas as pd
from classes import FeatureRegistry, FeatureStore


CHECKPOINT_DIR = '../Enrichement_models'
GROUPING_NAMES = ['tau_decaymode_2', 'njets']

DATA_PATH = '../data/dataframe_complete.feather'
MASKS_PATH = '../configs/masks.yaml'
VARIABLES_ENRICHMENT_PATH = '../configs/training_variables_enrichment.yaml'
NBINS = 40

FEATURE_STORE_DIR_WJETS = '../data/features/reduced_dataset/wjets'
FEATURE_REGISTRY_PATH_WJETS = '../data/features/reduced_dataset/wjets/feature_registry.json'
FEATURE_STORE_DIR_QCD = '../data/features/reduced_dataset/qcd'
FEATURE_REGISTRY_PATH_QCD = '../data/features/reduced_dataset/qcd/feature_registry.json'
DEFAULT_FEATURE_REGISTRY_PATH = '../data/features/feature_registry.json'

PROCESSES = ['diboson', 'DYjets', 'ST', 'ttbar', 'embedding']
PROCESSES_QCD = ['wjets', 'diboson', 'DYjets', 'ST', 'ttbar', 'embedding']



def equi_populated_bins(data, n_bins):

    data = np.asarray(data)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(data, quantiles)
    return bin_edges


def sum_process_histograms(
    df,
    bins,
    processes=PROCESSES,
    value_column='nn_output',
    weight_column=None,
):
    hist_sum = np.zeros(len(bins) - 1, dtype=np.float64)

    for process in processes:
        process_df = getattr(df.data, process).events
        values = process_df[value_column]
        weights = process_df[weight_column] if weight_column is not None else None
        hist_proc, _ = np.histogram(values, bins=bins, weights=weights)
        hist_sum += hist_proc

    return hist_sum


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


def reduced_data_wjets():
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    df = load_data(DATA_PATH, MASKS_PATH)
    training_variables = load_variables(VARIABLES_ENRICHMENT_PATH)

    for grouping in GROUPING_NAMES:
        checkpoint_dir = Path(CHECKPOINT_DIR) / "wjets"

        model = load_fold_combined_model(
            even_model_path=checkpoint_dir / grouping / 'fold_even',
            odd_model_path=checkpoint_dir / grouping / 'fold_odd'
        )

        region_view = df.data.DR_wjets
        region_df = region_view.events.copy()

        region_df["nn_output"] = _predict_fold_output(
            model=model,
            region_df=region_df,
            training_variables=training_variables,
            device=device,
        )

        region_view["nn_output"] = region_df["nn_output"].to_numpy()

        bins = equi_populated_bins(region_view["nn_output"], NBINS)

        hist_data, bin_edges = np.histogram(region_view["nn_output"], bins=bins)
        hist_qcd, _ = np.histogram(
            region_view["nn_output"],
            bins=bin_edges,
            weights=region_view[f"qcd_weight_{grouping}"],
        )
        hist_processes = sum_process_histograms(
            df,
            bins=bin_edges,
            processes=PROCESSES,
            value_column='nn_output',
        )

        reduced_weights = _calculate_scaled_event_weights_generalized(
            event_values=region_view["nn_output"].to_numpy(dtype=np.float32),
            event_original_weights=np.ones_like(region_view["nn_output"].to_numpy(dtype=np.float32)),
            bins=bin_edges,
            total_subtraction_per_bin=hist_qcd + hist_processes,
        )

        region_view[f'reduced_weight_{grouping}_nominal'] = reduced_weights

        feature_store_path = Path(FEATURE_STORE_DIR_WJETS) / f'reduced_weight_{grouping}.feather'

        registry = FeatureRegistry(FEATURE_REGISTRY_PATH_WJETS)
        store = FeatureStore(feature_store_path, registry)
        store.write(pd.DataFrame({
            "event": region_view["event"],
            f"reduced_weight_{grouping}_nominal": region_view[f"reduced_weight_{grouping}_nominal"],
        }))
        store.save()
        registry.save()

        default_registry = FeatureRegistry(DEFAULT_FEATURE_REGISTRY_PATH)
        default_registry.register([f"reduced_weight_{grouping}_nominal"], feature_store_path)
        default_registry.save()

    return df




def reduced_data_qcd():
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    df = load_data(DATA_PATH, MASKS_PATH)
    training_variables = load_variables(VARIABLES_ENRICHMENT_PATH)

    for grouping in GROUPING_NAMES:
        checkpoint_dir = Path(CHECKPOINT_DIR) / "qcd"

        model = load_fold_combined_model(
            even_model_path=checkpoint_dir / grouping / 'fold_even',
            odd_model_path=checkpoint_dir / grouping / 'fold_odd'
        )

        region_view = df.data.DR_qcd
        region_df = region_view.events.copy()

        region_df["nn_output"] = _predict_fold_output(
            model=model,
            region_df=region_df,
            training_variables=training_variables,
            device=device,
        )

        region_view["nn_output"] = region_df["nn_output"].to_numpy()

        bins = equi_populated_bins(region_view["nn_output"], NBINS)

        hist_data, bin_edges = np.histogram(region_view["nn_output"], bins=bins)

        hist_processes = sum_process_histograms(
            df,
            bins=bin_edges,
            processes=PROCESSES_QCD,
            value_column='nn_output',
        )

        reduced_weights = _calculate_scaled_event_weights_generalized(
            event_values=region_view["nn_output"].to_numpy(dtype=np.float32),
            event_original_weights=np.ones_like(region_view["nn_output"].to_numpy(dtype=np.float32)),
            bins=bin_edges,
            total_subtraction_per_bin=hist_processes,
        )

        region_view[f'reduced_weight_{grouping}_nominal'] = reduced_weights

        feature_store_path = Path(FEATURE_STORE_DIR_QCD) / f'reduced_weight_{grouping}.feather'

        registry = FeatureRegistry(FEATURE_REGISTRY_PATH_QCD)
        store = FeatureStore(feature_store_path, registry)
        store.write(pd.DataFrame({
            "event": region_view["event"],
            f"reduced_weight_{grouping}_nominal": region_view[f"reduced_weight_{grouping}_nominal"],
        }))
        store.save()
        registry.save()

        default_registry = FeatureRegistry(DEFAULT_FEATURE_REGISTRY_PATH)
        default_registry.register([f"reduced_weight_{grouping}_nominal"], feature_store_path)
        default_registry.save()

    return df






        
