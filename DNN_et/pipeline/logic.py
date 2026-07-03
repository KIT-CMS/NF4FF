import json
import law
import luigi
import yaml
from pathlib import Path

from BuildDataset import build_dataset
from BuildDataset_no_embedding import (
    REQUIRED_TRUE_TAU_PROCESSES,
    build_dataset as build_dataset_no_embedding,
)
from NF_training_njets import train_conditional_flows
from enrichment import (
    train_enrichment_qcd,
    train_enrichment_qcd_extrapolation,
    train_enrichment_qcd_fractions,
    train_enrichment_wjets,
)
from ReducedDataset import reduced_data_wjets, reduced_data_qcd
from plot_reduced_training_qcd import (
    create_qcd_extrapolation_training_plots,
    create_qcd_fraction_training_plots,
    create_qcd_training_plots,
)
from plot_reduced_training_wjets import create_wjets_training_plots
from training_squeezed_loss import (
    squeezing_label,
    train_squeezed_models,
)
from training_squeezed_loss_single_dnn import (
    train_squeezed_single_dnn_models,
)
from training_DRSR_squeezed_loss import (
    drsr_loss_limit_label,
    train_drsr_squeezed_models,
)
from drsr_corrections import (
    calculate_and_store_drsr_corrected_training_fraction_fake_factors,
    calculate_and_store_drsr_correction_factors,
    drsr_corrected_training_fraction_fake_factor_name,
    drsr_correction_name,
)
from multiclass_classification import (
    qcd_fraction_data_frame,
    train_fraction_classifier,
    validate_qcd_fraction_weights,
)
from plot_fractions import (
    calculate_and_store_fraction_nn_outputs,
    plot_fraction_comparisons,
)
from training_fraction_fake_factors import (
    calculate_and_store_extrapolation_corrected_training_fraction_fake_factors,
    calculate_and_store_corrected_training_fraction_fake_factors,
    calculate_and_store_training_fraction_fake_factors,
    corrected_training_fraction_fake_factor_name,
    extrapolation_corrected_training_fraction_fake_factor_name,
    training_fraction_fake_factor_name,
)
from ff_models_to_onnx import corrected_ff_models_to_onnx, ff_models_to_onnx
from single_dnn_workflow import (
    calculate_single_dnn_fake_factors,
    convert_single_dnn_models,
)
from taylor_coefficient_analysis import (
    TAYLOR_CATEGORIES,
    rewrite_taylor_plots,
    run_taylor_coefficient_categories,
    run_taylor_coefficient_comparison,
)
from ff_calculation import (
    calculate_and_store_classic_fake_factors,
    calculate_and_store_fake_factors,
)
from ff_model_uncertainty import (
    calculate_and_store_ff_dropout_mask_variation_features,
    calculate_and_store_ff_uncertainty_features,
)
from ff_gradient_covariance_uncertainty import (
    calculate_and_store_ff_gradient_covariance_dropout_mask_variation_features,
    calculate_and_store_ff_gradient_covariance_features,
)
from calculate_ff_corrected import (
    calculate_and_store_corrected_fake_factors,
)
from plotting import (
    create_corrected_fake_factor_closure_plots,
    create_drsr_correction_distribution_plots,
    create_extrapolation_corrected_mlf_closure_plots,
    create_fake_factor_opposite_grouping_distribution_plots,
    create_drsr_process_fake_factor_distribution_plots,
    create_fake_factor_plots,
    create_high_ff_closure_plots,
    create_high_fake_factor_distribution_plots,
    create_mlf_closure_plots,
    create_single_dnn_distribution_plots,
    create_single_dnn_fake_factor_plots,
)
from groupings import (
    GROUPING_NAMES,
    single_dnn_feature_suffix,
    squeezing_feature_suffix,
)
from uncertainty_model_workflow import (
    PROCESSES as UNCERTAINTY_PROCESSES,
    analyze_uncertainty_model_taylor_process,
    plot_uncertainty_model_taylor_process_normalized_single_order,
    plot_uncertainty_model_taylor_process_normalized_to_max,
    plot_uncertainty_model_taylor_process,
    save_uncertainty_combined_models,
    train_uncertainty_models,
    uncertainty_taylor_artifact_paths,
    uncertainty_taylor_coefficient_paths,
    uncertainty_taylor_normalized_artifact_paths,
    uncertainty_taylor_normalized_single_order_artifact_paths,
    uncertainty_taylor_normalized_single_order_plot_paths,
    uncertainty_taylor_normalized_plot_paths,
    uncertainty_taylor_plot_paths,
    write_uncertainty_taylor_normalized_manifest,
    write_uncertainty_taylor_normalized_single_order_manifest,
    write_uncertainty_taylor_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_ROOT = PROJECT_ROOT / 'Law_workflow_results'
WORKFLOW_DATA_ROOT = WORKFLOW_ROOT / 'data'
WORKFLOW_FEATURE_ROOT = WORKFLOW_DATA_ROOT / 'features'
ENRICHMENT_GROUPINGS = GROUPING_NAMES
FF_MODEL_OUTPUT_NAMES = {
    'wjets': 'Wjets',
    'qcd': 'QCD',
    'ttbar': 'ttbar',
}
TAYLOR_ARTIFACT_FILENAMES = {
    'json': 'taylor_coefficients.json',
    'metadata': 'metadata.json',
    'combined_png': 'taylor_coefficients_top_first_second.png',
    'combined_pdf': 'taylor_coefficients_top_first_second.pdf',
    'second_png': 'taylor_coefficients_top_second_order.png',
    'second_pdf': 'taylor_coefficients_top_second_order.pdf',
    'style': '.plot_style_v2_grouping_colors',
}


def taylor_category_scope(category):
    return 'full_dataset' if category == 'inclusive' else category


def ar_feature_file_is_complete(feature_path, column):
    """Check that a row-index keyed feature file covers all current AR rows."""
    import numpy as np
    import pandas as pd
    from classes import load_data

    feature_path = Path(feature_path)
    data_path = WORKFLOW_DATA_ROOT / 'dataframe_complete.feather'
    masks_path = PROJECT_ROOT / 'configs' / 'masks.yaml'
    if not feature_path.is_file() or not data_path.is_file():
        return False

    try:
        feature_frame = pd.read_feather(feature_path)
        if 'row_index' not in feature_frame.columns or column not in feature_frame.columns:
            return False

        df = load_data(data_path, masks_path)
        required_indices = df.events.index[df.mask('AR')]
        compact = (
            feature_frame[['row_index', column]]
            .drop_duplicates('row_index', keep='last')
            .set_index('row_index')
        )
        aligned = compact.reindex(required_indices)
        values = aligned[column].to_numpy(dtype=np.float64)
        return np.isfinite(values).all()
    except Exception:
        return False


class BuildDataset(law.Task):

    config_path = law.Parameter(default="../configs/root_data_path.yaml")

    def output(self):
        out_dir = WORKFLOW_DATA_ROOT
        return law.LocalFileTarget(out_dir / "dataframe_complete.feather")

    def run(self):

        df = build_dataset(self.config_path)

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)

        df.to_feather(self.output().path)

        print("BUILD OUTPUT:", self.output().path)


class BuildDatasetNoEmbedding(law.Task):
    """Build the no-embedding dataframe used by QCD extrapolation tasks."""

    config_path = law.Parameter(default="../configs/root_data_path.yaml")
    required_process_ids = frozenset(REQUIRED_TRUE_TAU_PROCESSES.values())
    schema_marker_name = ".schema_v1_no_embedding_true_tau_processes"

    def output(self):
        out_dir = WORKFLOW_DATA_ROOT
        return law.LocalFileTarget(
            out_dir / "dataframe_complete_no_embedding.feather"
        )

    def _schema_marker_path(self):
        return Path(self.output().path).parent / self.schema_marker_name

    def _missing_required_process_ids(self):
        import pandas as pd

        processes = set(
            pd.read_feather(
                self.output().path,
                columns=["process"],
            )["process"].unique()
        )
        return sorted(self.required_process_ids.difference(processes))

    def complete(self):
        if not super().complete():
            return False

        try:
            missing = self._missing_required_process_ids()
        except Exception:
            return False

        return not missing and self._schema_marker_path().is_file()

    def run(self):

        df = build_dataset_no_embedding(self.config_path)

        missing = sorted(
            self.required_process_ids.difference(set(df["process"].unique()))
        )
        if missing:
            raise RuntimeError(
                "BuildDatasetNoEmbedding did not produce the required true-tau "
                f"process ids {missing}. Check that diboson_T.root, "
                "DYjets_T.root, ST_T.root, and ttbar_T.root are present in "
                "the configured no-embedding data input directory."
            )

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)

        df.to_feather(self.output().path)
        self._schema_marker_path().write_text(
            "dataframe_complete_no_embedding.feather contains process ids 11, 12, 13, 14\n"
        )

        print("BUILD NO-EMBEDDING OUTPUT:", self.output().path)



class TrainEnrichmentProcess(law.Task):
    """Base task scaffold for per-process enrichment training."""

    process_name = "undefined"
    trainer = None

    def requires(self):
        return BuildDataset()

    config_model_path = law.Parameter(
        default="../configs/config_NN_enrichment.yaml"
    )

    def output(self):
        base = WORKFLOW_ROOT / "Enrichment_models" / self.process_name
        outputs = {
            f"{grouping}_{fold}": law.LocalDirectoryTarget(
                base / grouping / fold
            )
            for grouping in ENRICHMENT_GROUPINGS
            for fold in ('fold_even', 'fold_odd')
        }
        outputs["features_schema"] = law.LocalFileTarget(
            WORKFLOW_ROOT
            / "data"
            / "features"
            / self.process_name
            / ".row_index_schema_v1"
        )
        return outputs

    def run(self):
        if self.trainer is None:
            raise RuntimeError("No trainer function configured for this process task")

        print("TRAIN INPUT:", self.input().path)
        result = self.trainer(
            self.input().path,
            output_root=WORKFLOW_ROOT,
        )
        self.validate_training_result(result)
        schema_path = Path(self.output()["features_schema"].path)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text("row-index keyed enrichment features\n")
        print("TRAIN OUTPUT:", result["combined_model_path"])

    def validate_training_result(self, result):
        pass


class TrainEnrichmentWjetsV2(TrainEnrichmentProcess):
    process_name = "wjets"
    trainer = staticmethod(train_enrichment_wjets)


class TrainEnrichmentQCDV2(TrainEnrichmentProcess):
    process_name = "qcd"
    trainer = staticmethod(train_enrichment_qcd)


class TrainEnrichmentQCDFractions(TrainEnrichmentProcess):
    """Train QCD enrichment weights in the fractions determination region."""

    process_name = "qcd_fraction"
    trainer = staticmethod(train_enrichment_qcd_fractions)

    def output(self):
        outputs = super().output()
        outputs["features_schema"] = law.LocalFileTarget(
            WORKFLOW_FEATURE_ROOT
            / self.process_name
            / ".row_index_schema_v3_qcd_weights_ss_no_nan"
        )
        outputs.update({
            f"feature_{grouping}": law.LocalFileTarget(
                WORKFLOW_FEATURE_ROOT
                / self.process_name
                / f"qcd_fraction_weights_{grouping}.feather"
            )
            for grouping in ENRICHMENT_GROUPINGS
        })
        return outputs

    def validate_training_result(self, result):
        from classes import load_data

        df = load_data(
            WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            PROJECT_ROOT / 'configs' / 'masks.yaml',
        )
        validate_qcd_fraction_weights(qcd_fraction_data_frame(df))


class TrainEnrichmentQCDExtrapolation(TrainEnrichmentProcess):
    """Train QCD extrapolation weights for same-sign AR and SR regions."""

    process_name = "qcd_extrapolation"
    trainer = staticmethod(train_enrichment_qcd_extrapolation)

    def requires(self):
        return BuildDatasetNoEmbedding()

    def output(self):
        outputs = super().output()
        outputs["features_schema"] = law.LocalFileTarget(
            WORKFLOW_FEATURE_ROOT
            / self.process_name
            / ".row_index_schema_v1_no_embedding"
        )
        outputs.update({
            f"feature_{grouping}": law.LocalFileTarget(
                WORKFLOW_FEATURE_ROOT
                / self.process_name
                / f"qcd_extrapolation_weights_{grouping}.feather"
            )
            for grouping in ENRICHMENT_GROUPINGS
        })
        return outputs

    def validate_training_result(self, result):
        from classes import load_data_no_embedding
        import numpy as np

        weight_column = "weight_qcd_extrapolation_njets"
        df = load_data_no_embedding(
            WORKFLOW_DATA_ROOT / 'dataframe_complete_no_embedding.feather',
            PROJECT_ROOT / 'configs' / 'masks.yaml',
            feature_registry_path=(
                WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
            ),
        )
        df[weight_column]
        for region_name in ("AR_SS", "SR_SS"):
            frame = getattr(df.data, region_name).events
            finite = np.isfinite(
                frame[weight_column].to_numpy(dtype=np.float64)
            )
            if not finite.all():
                raise ValueError(
                    f"{weight_column} contains non-finite weights for "
                    f"{int((~finite).sum())}/{len(frame)} "
                    f"data {region_name} events."
                )


class ReducedDataset(law.Task):
    """Compute reduced datasets for both W+jets and QCD processes."""

    def requires(self):
        return {
            'wjets': TrainEnrichmentWjetsV2(),
            'qcd': TrainEnrichmentQCDV2(),
        }

    def output(self):
        return {
            'wjets': law.LocalDirectoryTarget(WORKFLOW_FEATURE_ROOT / 'reduced_dataset' / 'wjets'),
            'qcd': law.LocalDirectoryTarget(WORKFLOW_FEATURE_ROOT / 'reduced_dataset' / 'qcd'),
            'schema': law.LocalFileTarget(
                WORKFLOW_FEATURE_ROOT / 'reduced_dataset' / '.schema_v7_groupings'
            ),
        }

    def run(self):
        reduced_data_wjets(output_root=WORKFLOW_ROOT)
        reduced_data_qcd(output_root=WORKFLOW_ROOT)
        schema_path = Path(self.output()['schema'].path)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(
            "explicit full-process inference and data-only reduced weights\n"
        )
        print("REDUCED DATASET WJETS OUTPUT:", self.output()['wjets'].path)
        print("REDUCED DATASET QCD OUTPUT:", self.output()['qcd'].path)


class PlotTrainingResultsWjets(law.Task):
    """Plot W+jets enrichment diagnostics for both grouping choices."""

    n_bins = luigi.IntParameter(default=20)

    def requires(self):
        return ReducedDataset()

    def output(self):
        output_dir = WORKFLOW_ROOT / 'plots' / 'enrichment_wjets'
        return {
            f"{plot_name}_{grouping}_{extension}": law.LocalFileTarget(
                output_dir / f"{plot_name}_wjets_{grouping}.{extension}"
            )
            for grouping in ENRICHMENT_GROUPINGS
            for plot_name in ('training_composition', 'reduced_closure')
            for extension in ('png', 'pdf')
        }

    def run(self):
        create_wjets_training_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            output_dir=WORKFLOW_ROOT / 'plots' / 'enrichment_wjets',
            qcd_weight_store_dir=WORKFLOW_FEATURE_ROOT / 'wjets',
            n_bins=self.n_bins,
        )


class PlotTrainingResultsQCD(law.Task):
    """Plot QCD enrichment diagnostics for both grouping choices."""

    n_bins = luigi.IntParameter(default=20)

    def requires(self):
        return ReducedDataset()

    def output(self):
        output_dir = WORKFLOW_ROOT / 'plots' / 'enrichment_qcd'
        return {
            f"{plot_name}_{grouping}_{extension}": law.LocalFileTarget(
                output_dir / f"{plot_name}_qcd_{grouping}.{extension}"
            )
            for grouping in ENRICHMENT_GROUPINGS
            for plot_name in ('training_composition', 'reduced_closure')
            for extension in ('png', 'pdf')
        }

    def run(self):
        create_qcd_training_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            output_dir=WORKFLOW_ROOT / 'plots' / 'enrichment_qcd',
            reduced_weight_store_dir=(
                WORKFLOW_FEATURE_ROOT / 'reduced_dataset' / 'qcd'
            ),
            n_bins=self.n_bins,
        )


class PlotTrainingResultsQCD2(law.Task):
    """Plot enrichment diagnostics in the QCD fractions region."""

    n_bins = luigi.IntParameter(default=20)

    def requires(self):
        return TrainEnrichmentQCDFractions()

    def output(self):
        output_dir = WORKFLOW_ROOT / 'plots' / 'enrichment_qcd2'
        return {
            f"{plot_name}_{grouping}_{extension}": law.LocalFileTarget(
                output_dir / f"{plot_name}_qcd2_{grouping}.{extension}"
            )
            for grouping in ENRICHMENT_GROUPINGS
            for plot_name in ('training_composition', 'reduced_closure')
            for extension in ('png', 'pdf')
        }

    def run(self):
        create_qcd_fraction_training_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            output_dir=WORKFLOW_ROOT / 'plots' / 'enrichment_qcd2',
            model_dir=WORKFLOW_ROOT / 'Enrichment_models' / 'qcd_fraction',
            qcd_weight_store_dir=WORKFLOW_FEATURE_ROOT / 'qcd_fraction',
            training_variables_path=(
                PROJECT_ROOT
                / 'configs'
                / 'training_variables_enrichment.yaml'
            ),
            n_bins=self.n_bins,
        )


class PlotTrainingResultsQCDExtrapolation(law.Task):
    """Plot QCD enrichment diagnostics in DR_qcd_extrapolation."""

    n_bins = luigi.IntParameter(default=20)

    def requires(self):
        return TrainEnrichmentQCDExtrapolation()

    def output(self):
        output_dir = WORKFLOW_ROOT / 'plots' / 'enrichment_qcd_extrapolation'
        return {
            f"{plot_name}_{grouping}_{extension}": law.LocalFileTarget(
                output_dir
                / f"{plot_name}_qcd_extrapolation_{grouping}.{extension}"
            )
            for grouping in ENRICHMENT_GROUPINGS
            for plot_name in ('training_composition', 'reduced_closure')
            for extension in ('png', 'pdf')
        }

    def run(self):
        create_qcd_extrapolation_training_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete_no_embedding.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            output_dir=WORKFLOW_ROOT / 'plots' / 'enrichment_qcd_extrapolation',
            model_dir=(
                WORKFLOW_ROOT / 'Enrichment_models' / 'qcd_extrapolation'
            ),
            qcd_weight_store_dir=(
                WORKFLOW_FEATURE_ROOT / 'qcd_extrapolation'
            ),
            training_variables_path=(
                PROJECT_ROOT
                / 'configs'
                / 'training_variables_enrichment.yaml'
            ),
            n_bins=self.n_bins,
        )


class TrainFractionClassifier(law.Task):
    """Train the three-class fraction classifier."""

    def requires(self):
        return TrainEnrichmentQCDFractions()

    def output(self):
        output_dir = WORKFLOW_ROOT / 'training_fraction'
        return {
            'fold_even': law.LocalFileTarget(
                output_dir / 'fold_even' / 'model_weights.pth'
            ),
            'fold_odd': law.LocalFileTarget(
                output_dir / 'fold_odd' / 'model_weights.pth'
            ),
            'combined': law.LocalFileTarget(
                output_dir / 'model_weights.pth'
            ),
        }

    def run(self):
        output_dir = WORKFLOW_ROOT / 'training_fraction'
        train_fraction_classifier(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_var_path=PROJECT_ROOT / 'configs' / 'training_variables.yaml',
            output_dir=output_dir,
        )

        print("FRACTION CLASSIFIER OUTPUT:", output_dir)


class PlotFractions(law.Task):
    """Evaluate and plot the three-class process-fraction classifier."""

    process_fractions_path = law.Parameter(
        default=(
            '/work/mmoser/TauFakeFactors.back/workdir/'
            'ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz'
        )
    )
    batch_size = luigi.IntParameter(default=100_000)

    def requires(self):
        return TrainFractionClassifier()

    def _variables_small(self):
        config_path = PROJECT_ROOT / 'configs' / 'plotting.yaml'
        with open(config_path, 'r', encoding='utf-8') as stream:
            config = yaml.safe_load(stream) or {}
        return tuple(config.get('variables_set_small', ()))

    def output(self):
        plot_dir = WORKFLOW_ROOT / 'plots' / 'training_fraction'
        feature_path = (
            WORKFLOW_FEATURE_ROOT
            / 'training_fraction'
            / 'process_fractions.feather'
        )
        outputs = {
            'features': law.LocalFileTarget(feature_path),
            'schema': law.LocalFileTarget(
                WORKFLOW_FEATURE_ROOT
                / 'training_fraction'
                / '.schema_v2_all_ar'
            ),
        }
        outputs.update({
            f'{variable}_{extension}': law.LocalFileTarget(
                plot_dir / f'training_fraction_{variable}.{extension}'
            )
            for variable in self._variables_small()
            for extension in ('png', 'pdf')
        })
        return outputs

    def run(self):
        feature_path = calculate_and_store_fraction_nn_outputs(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_var_path=PROJECT_ROOT / 'configs' / 'training_variables.yaml',
            model_dir=WORKFLOW_ROOT / 'training_fraction',
            feature_store_path=(
                WORKFLOW_FEATURE_ROOT
                / 'training_fraction'
                / 'process_fractions.feather'
            ),
            feature_registry_path=WORKFLOW_FEATURE_ROOT / 'feature_registry.json',
            batch_size=self.batch_size,
        )

        from classes import load_data

        df = load_data(
            WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            PROJECT_ROOT / 'configs' / 'masks.yaml',
            feature_registry_path=WORKFLOW_FEATURE_ROOT / 'feature_registry.json',
        )
        for column in ('fraction_qcd', 'fraction_wjets', 'fraction_ttbar'):
            df[column]
        plot_fraction_comparisons(
            df.data.AR.events,
            output_dir=WORKFLOW_ROOT / 'plots' / 'training_fraction',
            process_fractions_path=self.process_fractions_path,
            plotting_config_path=PROJECT_ROOT / 'configs' / 'plotting.yaml',
            labels_config_path=PROJECT_ROOT / 'configs' / 'labels.yaml',
        )

        print("FRACTION FEATURE OUTPUT:", feature_path)
        print("FRACTION PLOT OUTPUT:", WORKFLOW_ROOT / 'plots' / 'training_fraction')
        Path(self.output()['schema'].path).write_text(
            "fraction_qcd, fraction_wjets, fraction_ttbar calculated for all AR rows\n"
        )


class TrainSqueezedModels(law.Task):
    """Train grouped FF models with an optional output squeezing probability."""

    squeezing = luigi.OptionalFloatParameter(default=None)

    def requires(self):
        return ReducedDataset()

    def output(self):
        return law.LocalFileTarget(
            WORKFLOW_ROOT
            / 'Training_results_squeezed'
            / squeezing_label(self.squeezing)
            / '.groupings_v2_complete'
        )

    def run(self):
        train_squeezed_models(
            squeezing=self.squeezing,
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_var_path=PROJECT_ROOT / 'configs' / 'training_variables.yaml',
            nn_config_path=PROJECT_ROOT / 'configs' / 'DNN.yaml',
            checkpoint_dir=WORKFLOW_ROOT / 'Training_results_squeezed',
            reduced_weight_dir=WORKFLOW_FEATURE_ROOT / 'reduced_dataset',
        )

        output_path = Path(self.output().path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            f"squeezing_probability={self.squeezing}\n"
        )


class CalculateGroupedDNNTaylorCoefficients(law.Task):
    """Calculate Taylor coefficients for all grouped DNN process models."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    max_order = luigi.IntParameter(default=2)
    top_n = luigi.IntParameter(default=10)
    batch_size = luigi.IntParameter(default=1024, significant=False)

    def requires(self):
        return TrainSqueezedModels(squeezing=self.squeezing)

    def output(self):
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'grouped_dnn'
            / squeezing_label(self.squeezing)
            / f'order_{self.max_order}'
        )
        return {
            f'{category}_{process}_{grouping}_{artifact}': law.LocalFileTarget(
                output_root
                / taylor_category_scope(category)
                / f'top_{self.top_n}'
                / grouping
                / process
                / filename
            )
            for category, _ in TAYLOR_CATEGORIES
            for process in ('wjets', 'qcd', 'ttbar')
            for grouping in ENRICHMENT_GROUPINGS
            for artifact, filename in TAYLOR_ARTIFACT_FILENAMES.items()
        }

    def run(self):
        trained_models_dir = Path(self.input().path).parent
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'grouped_dnn'
            / squeezing_label(self.squeezing)
            / f'order_{self.max_order}'
        )
        for grouping in ENRICHMENT_GROUPINGS:
            for process in ('wjets', 'qcd', 'ttbar'):
                model_dir = trained_models_dir / grouping / process
                run_taylor_coefficient_categories(
                    even_model_path=model_dir / 'fold_even',
                    odd_model_path=model_dir / 'fold_odd',
                    data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
                    masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
                    training_var_path=(
                        PROJECT_ROOT / 'configs' / 'training_variables.yaml'
                    ),
                    process=process,
                    output_dirs={
                        category: (
                            output_root
                            / taylor_category_scope(category)
                            / f'top_{self.top_n}'
                            / grouping
                            / process
                        )
                        for category, _ in TAYLOR_CATEGORIES
                    },
                    analysis_label=grouping,
                    model_type='GroupedDNN',
                    max_order=self.max_order,
                    top_n=self.top_n,
                    batch_size=self.batch_size,
                )


class PlotGroupedDNNTaylorCoefficients(law.Task):
    """Regenerate grouped-DNN Taylor plots from existing coefficient JSONs."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    max_order = luigi.IntParameter(default=2)
    top_n = luigi.IntParameter(default=10)

    def output(self):
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'grouped_dnn'
            / squeezing_label(self.squeezing)
            / f'order_{self.max_order}'
        )
        return law.LocalFileTarget(
            output_root / f'plot_only_manifest_top_{self.top_n}.json'
        )

    def run(self):
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'grouped_dnn'
            / squeezing_label(self.squeezing)
            / f'order_{self.max_order}'
        )
        coefficient_paths = sorted(
            output_root.glob(
                f'*/top_{self.top_n}/*/*/taylor_coefficients.json'
            )
        )
        if not coefficient_paths:
            raise FileNotFoundError(
                "No grouped-DNN Taylor coefficient JSON files found under "
                f"{output_root} for top_n={self.top_n}."
            )

        rewritten = []
        for coefficient_path in coefficient_paths:
            paths = rewrite_taylor_plots(
                coefficient_path.parent,
                top_n=self.top_n,
            )
            rewritten.append({
                key: str(path)
                for key, path in paths.items()
                if key not in ('json', 'metadata')
            })

        manifest_path = Path(self.output().path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(rewritten, indent=2) + "\n")


class TrainUncertaintyModels(law.Task):
    """Train 100 full-dataset njets models for each process."""

    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)

    def requires(self):
        return ReducedDataset()

    def output(self):
        return law.LocalFileTarget(
            WORKFLOW_ROOT
            / 'Training_results_uncertainties'
            / f'seeds_{self.seed_start}_{self.seed_end}'
            / 'training_manifest.json'
        )

    def run(self):
        train_uncertainty_models(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_var_path=(
                PROJECT_ROOT / 'configs' / 'training_variables.yaml'
            ),
            nn_config_path=PROJECT_ROOT / 'configs' / 'DNN.yaml',
            output_dir=Path(self.output().path).parent,
            reduced_weight_dir=(
                WORKFLOW_FEATURE_ROOT / 'reduced_dataset'
            ),
            seed_start=self.seed_start,
            seed_end=self.seed_end,
        )


class SaveUncertaintyCombinedModels(law.Task):
    """Save likelihood-ratio FF models for the uncertainty ensemble."""

    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)

    def requires(self):
        return TrainUncertaintyModels(
            seed_start=self.seed_start,
            seed_end=self.seed_end,
        )

    def output(self):
        output_dir = (
            WORKFLOW_ROOT
            / 'CombinedModelsUncertainties'
            / f'seeds_{self.seed_start}_{self.seed_end}'
        )
        outputs = {
            f"{process}_{seed}_torch": law.LocalFileTarget(
                output_dir
                / FF_MODEL_OUTPUT_NAMES[process]
                / 'njets'
                / str(seed)
                / 'torch_model'
                / 'model_weights.pth'
            )
            for process in ('wjets', 'qcd', 'ttbar')
            for seed in range(self.seed_start, self.seed_end + 1)
        }
        outputs['manifest'] = law.LocalFileTarget(
            output_dir / 'combined_models_manifest.json'
        )
        return outputs

    def run(self):
        output_dir = (
            WORKFLOW_ROOT
            / 'CombinedModelsUncertainties'
            / f'seeds_{self.seed_start}_{self.seed_end}'
        )
        save_uncertainty_combined_models(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            trained_models_dir=Path(self.input().path).parent,
            reduced_weight_dir=(
                WORKFLOW_FEATURE_ROOT / 'reduced_dataset'
            ),
            output_dir=output_dir,
            seed_start=self.seed_start,
            seed_end=self.seed_end,
        )


class CalculateFakeFactorModelUncertaintyProcess(law.Task):
    """Calculate and store FF uncertainty ensemble features for one process."""

    process = luigi.ChoiceParameter(choices=UNCERTAINTY_PROCESSES)
    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)

    def requires(self):
        return SaveUncertaintyCombinedModels(
            seed_start=self.seed_start,
            seed_end=self.seed_end,
        )

    def output(self):
        output_dir = (
            WORKFLOW_FEATURE_ROOT
            / 'fake_factor_model_uncertainty'
            / self.process
            / f'seeds_{self.seed_start}_{self.seed_end}'
        )
        return {
            'features': law.LocalFileTarget(
                output_dir / 'fake_factor_model_uncertainty.feather'
            ),
            'schema': law.LocalFileTarget(
                output_dir / '.schema_v1_njets_100_models'
            ),
        }

    def run(self):
        calculate_and_store_ff_uncertainty_features(
            process=self.process,
            feature_store_path=self.output()['features'].path,
            feature_registry_path=(
                WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
            ),
            seeds=range(self.seed_start, self.seed_end + 1),
            combined_models_dir=(
                WORKFLOW_ROOT
                / 'CombinedModelsUncertainties'
                / f'seeds_{self.seed_start}_{self.seed_end}'
            ),
            overwrite=False,
        )
        Path(self.output()['schema'].path).write_text(
            f"FF_{self.process}_down, FF_{self.process}_nominal, "
            f"FF_{self.process}_up, and FF_{self.process}_0..99 for njets\n"
        )


class CalculateFakeFactorModelUncertainty(law.Task):
    """Calculate FF uncertainty ensemble features for all processes."""

    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)

    def requires(self):
        return {
            process: CalculateFakeFactorModelUncertaintyProcess(
                process=process,
                seed_start=self.seed_start,
                seed_end=self.seed_end,
            )
            for process in UNCERTAINTY_PROCESSES
        }

    def output(self):
        output_dir = (
            WORKFLOW_FEATURE_ROOT
            / 'fake_factor_model_uncertainty'
            / f'seeds_{self.seed_start}_{self.seed_end}'
        )
        return law.LocalFileTarget(output_dir / 'all_processes_manifest.txt')

    def run(self):
        output_path = Path(self.output().path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{process}: {self.input()[process]['features'].path}"
            for process in UNCERTAINTY_PROCESSES
        ]
        output_path.write_text("\n".join(lines) + "\n")


class CalculateWjetsFakeFactorModelUncertainty(
    CalculateFakeFactorModelUncertaintyProcess
):
    """Backward-compatible W+jets-only task name."""

    process = 'wjets'


class CalculateFakeFactorDropoutMaskVariation(law.Task):
    """Calculate FF features from 100 fixed dropout masks of seed-100 W+jets."""

    model_seed = luigi.IntParameter(default=100)
    n_dropout_masks = luigi.IntParameter(default=100)
    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)

    def requires(self):
        return SaveUncertaintyCombinedModels(
            seed_start=self.seed_start,
            seed_end=self.seed_end,
        )

    def output(self):
        output_dir = (
            WORKFLOW_FEATURE_ROOT
            / 'fake_factor_dropout_mask_variation'
            / f'seed_{self.model_seed}'
            / f'n_masks_{self.n_dropout_masks}'
        )
        return {
            'features': law.LocalFileTarget(
                output_dir / 'fake_factor_dropout_mask_variation.feather'
            ),
            'schema': law.LocalFileTarget(
                output_dir / '.schema_v1_dmv'
            ),
        }

    def run(self):
        calculate_and_store_ff_dropout_mask_variation_features(
            process='wjets',
            feature_store_path=self.output()['features'].path,
            feature_registry_path=(
                WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
            ),
            model_seed=self.model_seed,
            n_masks=self.n_dropout_masks,
            combined_models_dir=(
                WORKFLOW_ROOT
                / 'CombinedModelsUncertainties'
                / f'seeds_{self.seed_start}_{self.seed_end}'
            ),
            overwrite=False,
        )
        Path(self.output()['schema'].path).write_text(
            "FF_nominal_dmv, FF_up_dmv, FF_down_dmv, "
            f"and FF_0_dmv..FF_{self.n_dropout_masks - 1}_dmv "
            f"from W+jets seed {self.model_seed} dropout masks\n"
        )


class CalculateWjetsGradientCovarianceUncertainty(law.Task):
    """Propagate W+jets/njets input covariance through the 100 FF models."""

    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)
    batch_size = luigi.IntParameter(default=2048, significant=False)
    overwrite = luigi.BoolParameter(default=False, significant=False)

    def requires(self):
        return SaveUncertaintyCombinedModels(
            seed_start=self.seed_start,
            seed_end=self.seed_end,
        )

    def output(self):
        output_dir = (
            WORKFLOW_FEATURE_ROOT
            / 'fake_factor_gradient_covariance_uncertainty'
            / 'wjets'
            / f'seeds_{self.seed_start}_{self.seed_end}'
        )
        return {
            'features': law.LocalFileTarget(
                output_dir / 'fake_factor_gradient_covariance_uncertainty.feather'
            ),
            'covariances': law.LocalDirectoryTarget(
                output_dir / 'covariances'
            ),
            'schema': law.LocalFileTarget(
                output_dir / '.schema_v1_wjets_njets_100_models'
            ),
        }

    def run(self):
        calculate_and_store_ff_gradient_covariance_features(
            process='wjets',
            grouping='njets',
            feature_store_path=self.output()['features'].path,
            feature_registry_path=(
                WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
            ),
            seeds=range(self.seed_start, self.seed_end + 1),
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_variables_path=(
                PROJECT_ROOT / 'configs' / 'training_variables.yaml'
            ),
            reduced_weight_dir=(
                WORKFLOW_FEATURE_ROOT / 'reduced_dataset'
            ),
            combined_models_dir=(
                WORKFLOW_ROOT
                / 'CombinedModelsUncertainties'
                / f'seeds_{self.seed_start}_{self.seed_end}'
            ),
            batch_size=self.batch_size,
            covariance_output_dir=self.output()['covariances'].path,
            overwrite=self.overwrite,
        )
        Path(self.output()['schema'].path).write_text(
            "FF_wjets_gradcov_sigma_sum, FF_wjets_gradcov_sigma_mean, "
            "FF_wjets_gradcov_variance_sum, "
            "FF_wjets_gradcov_variance_mean for njets\n"
        )


class CalculateWjetsGradientCovarianceDropoutMaskVariation(law.Task):
    """Propagate W+jets/njets covariance through fixed dropout masks."""

    model_seed = luigi.IntParameter(default=100)
    n_dropout_masks = luigi.IntParameter(default=100)
    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)
    batch_size = luigi.IntParameter(default=2048, significant=False)
    overwrite = luigi.BoolParameter(default=False, significant=False)

    def requires(self):
        return SaveUncertaintyCombinedModels(
            seed_start=self.seed_start,
            seed_end=self.seed_end,
        )

    def output(self):
        output_dir = (
            WORKFLOW_FEATURE_ROOT
            / 'fake_factor_gradient_covariance_dropout_mask_variation'
            / 'wjets'
            / f'seed_{self.model_seed}'
            / f'n_masks_{self.n_dropout_masks}'
        )
        return {
            'features': law.LocalFileTarget(
                output_dir
                / 'fake_factor_gradient_covariance_dropout_mask_variation.feather'
            ),
            'covariances': law.LocalDirectoryTarget(
                output_dir / 'covariances'
            ),
            'schema': law.LocalFileTarget(
                output_dir / '.schema_v1_wjets_njets_dmv'
            ),
        }

    def run(self):
        calculate_and_store_ff_gradient_covariance_dropout_mask_variation_features(
            process='wjets',
            grouping='njets',
            feature_store_path=self.output()['features'].path,
            feature_registry_path=(
                WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
            ),
            model_seed=self.model_seed,
            n_masks=self.n_dropout_masks,
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_variables_path=(
                PROJECT_ROOT / 'configs' / 'training_variables.yaml'
            ),
            reduced_weight_dir=(
                WORKFLOW_FEATURE_ROOT / 'reduced_dataset'
            ),
            combined_models_dir=(
                WORKFLOW_ROOT
                / 'CombinedModelsUncertainties'
                / f'seeds_{self.seed_start}_{self.seed_end}'
            ),
            batch_size=self.batch_size,
            covariance_output_dir=self.output()['covariances'].path,
            overwrite=self.overwrite,
        )
        Path(self.output()['schema'].path).write_text(
            "FF_wjets_gradcov_sigma_sum_dmv, "
            "FF_wjets_gradcov_sigma_mean_dmv, "
            "FF_wjets_gradcov_variance_sum_dmv, "
            "FF_wjets_gradcov_variance_mean_dmv for njets dropout masks\n"
        )


class CalculateUncertaintyModelTaylorCoefficientsProcess(law.Task):
    """Calculate uncertainty Taylor coefficients for one process."""

    process = luigi.ChoiceParameter(choices=UNCERTAINTY_PROCESSES)
    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)
    dropout_model_seed = luigi.IntParameter(default=100)
    n_dropout_masks = luigi.IntParameter(default=100)
    max_order = luigi.IntParameter(default=2)
    top_n = luigi.IntParameter(default=10, significant=False)
    batch_size = luigi.IntParameter(default=4096, significant=False)
    cpu_threads = luigi.IntParameter(default=8, significant=False)

    def requires(self):
        return TrainUncertaintyModels(
            seed_start=self.seed_start,
            seed_end=self.seed_end,
        )

    def output(self):
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'uncertainty_models'
            / f'seeds_{self.seed_start}_{self.seed_end}'
            / f'order_{self.max_order}'
        )
        return {
            key: law.LocalFileTarget(path)
            for key, path in uncertainty_taylor_coefficient_paths(
                output_root,
                self.process,
            ).items()
        }

    def run(self):
        analyze_uncertainty_model_taylor_process(
            process=self.process,
            models_dir=Path(self.input().path).parent,
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_var_path=(
                PROJECT_ROOT / 'configs' / 'training_variables.yaml'
            ),
            output_dir=(
                WORKFLOW_ROOT
                / 'Taylor_coefficient_analysis'
                / 'uncertainty_models'
                / f'seeds_{self.seed_start}_{self.seed_end}'
                / f'order_{self.max_order}'
            ),
            seed_start=self.seed_start,
            seed_end=self.seed_end,
            dropout_model_seed=self.dropout_model_seed,
            n_dropout_masks=self.n_dropout_masks,
            max_order=self.max_order,
            top_n=self.top_n,
            batch_size=self.batch_size,
            cpu_threads=self.cpu_threads,
        )


class CalculateUncertaintyModelTaylorCoefficients(law.Task):
    """Calculate uncertainty Taylor coefficients in parallel by process."""

    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)
    dropout_model_seed = luigi.IntParameter(default=100)
    n_dropout_masks = luigi.IntParameter(default=100)
    max_order = luigi.IntParameter(default=2)
    top_n = luigi.IntParameter(default=10, significant=False)
    batch_size = luigi.IntParameter(default=4096, significant=False)
    cpu_threads = luigi.IntParameter(default=8, significant=False)

    def requires(self):
        return {
            process: CalculateUncertaintyModelTaylorCoefficientsProcess(
                process=process,
                seed_start=self.seed_start,
                seed_end=self.seed_end,
                dropout_model_seed=self.dropout_model_seed,
                n_dropout_masks=self.n_dropout_masks,
                max_order=self.max_order,
                top_n=self.top_n,
                batch_size=self.batch_size,
                cpu_threads=self.cpu_threads,
            )
            for process in UNCERTAINTY_PROCESSES
        }

    def output(self):
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'uncertainty_models'
            / f'seeds_{self.seed_start}_{self.seed_end}'
            / f'order_{self.max_order}'
        )
        return {
            f'{process}_{key}': law.LocalFileTarget(path)
            for process in UNCERTAINTY_PROCESSES
            for key, path in uncertainty_taylor_coefficient_paths(
                output_root,
                process,
            ).items()
        }

    def run(self):
        pass


class CalculateFirstOrderUncertaintyModelTaylorCoefficients(
    CalculateUncertaintyModelTaylorCoefficients
):
    """Calculate first-order uncertainty Taylor coefficients only."""

    max_order = luigi.IntParameter(default=1)


class PlotUncertaintyModelTaylorCoefficientsProcess(law.Task):
    """Summarize and plot calculated Taylor coefficients for one process."""

    process = luigi.ChoiceParameter(choices=UNCERTAINTY_PROCESSES)
    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)
    dropout_model_seed = luigi.IntParameter(default=100)
    n_dropout_masks = luigi.IntParameter(default=100)
    max_order = luigi.IntParameter(default=2)
    top_n = luigi.IntParameter(default=10)
    batch_size = luigi.IntParameter(default=4096, significant=False)
    cpu_threads = luigi.IntParameter(default=8, significant=False)

    def requires(self):
        return CalculateUncertaintyModelTaylorCoefficientsProcess(
            process=self.process,
            seed_start=self.seed_start,
            seed_end=self.seed_end,
            dropout_model_seed=self.dropout_model_seed,
            n_dropout_masks=self.n_dropout_masks,
            max_order=self.max_order,
            top_n=self.top_n,
            batch_size=self.batch_size,
            cpu_threads=self.cpu_threads,
        )

    def output(self):
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'uncertainty_models'
            / f'seeds_{self.seed_start}_{self.seed_end}'
            / f'order_{self.max_order}'
        )
        return {
            key: law.LocalFileTarget(path)
            for key, path in uncertainty_taylor_plot_paths(
                output_root,
                self.process,
                self.top_n,
            ).items()
        }

    def run(self):
        plot_uncertainty_model_taylor_process(
            process=self.process,
            output_dir=(
                WORKFLOW_ROOT
                / 'Taylor_coefficient_analysis'
                / 'uncertainty_models'
                / f'seeds_{self.seed_start}_{self.seed_end}'
                / f'order_{self.max_order}'
            ),
            top_n=self.top_n,
        )


class PlotUncertaintyModelTaylorCoefficients(law.Task):
    """Plot uncertainty Taylor results without recalculating coefficients."""

    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)
    dropout_model_seed = luigi.IntParameter(default=100)
    n_dropout_masks = luigi.IntParameter(default=100)
    max_order = luigi.IntParameter(default=2)
    top_n = luigi.IntParameter(default=10)
    batch_size = luigi.IntParameter(default=4096, significant=False)
    cpu_threads = luigi.IntParameter(default=8, significant=False)

    def requires(self):
        return {
            process: PlotUncertaintyModelTaylorCoefficientsProcess(
                process=process,
                seed_start=self.seed_start,
                seed_end=self.seed_end,
                dropout_model_seed=self.dropout_model_seed,
                n_dropout_masks=self.n_dropout_masks,
                max_order=self.max_order,
                top_n=self.top_n,
                batch_size=self.batch_size,
                cpu_threads=self.cpu_threads,
            )
            for process in UNCERTAINTY_PROCESSES
        }

    def output(self):
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'uncertainty_models'
            / f'seeds_{self.seed_start}_{self.seed_end}'
            / f'order_{self.max_order}'
        )
        outputs = {
            f'{process}_{key}': law.LocalFileTarget(path)
            for process in UNCERTAINTY_PROCESSES
            for key, path in uncertainty_taylor_artifact_paths(
                output_root,
                process,
                self.top_n,
            ).items()
        }
        outputs['manifest'] = law.LocalFileTarget(
            output_root / 'analysis_manifest.json'
        )
        return outputs

    def run(self):
        write_uncertainty_taylor_manifest(
            output_dir=Path(self.output()['manifest'].path).parent,
            seed_start=self.seed_start,
            seed_end=self.seed_end,
            dropout_model_seed=self.dropout_model_seed,
            n_dropout_masks=self.n_dropout_masks,
            max_order=self.max_order,
            top_n=self.top_n,
            batch_size=self.batch_size,
            cpu_threads=self.cpu_threads,
        )


class PlotNormalizedUncertaintyModelTaylorCoefficientsProcess(law.Task):
    """Plot uncertainty Taylor coefficients normalized per model/mask."""

    process = luigi.ChoiceParameter(choices=UNCERTAINTY_PROCESSES)
    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)
    dropout_model_seed = luigi.IntParameter(default=100)
    n_dropout_masks = luigi.IntParameter(default=100)
    max_order = luigi.IntParameter(default=2)
    top_n = luigi.IntParameter(default=10)
    batch_size = luigi.IntParameter(default=4096, significant=False)
    cpu_threads = luigi.IntParameter(default=8, significant=False)

    def requires(self):
        return CalculateUncertaintyModelTaylorCoefficientsProcess(
            process=self.process,
            seed_start=self.seed_start,
            seed_end=self.seed_end,
            dropout_model_seed=self.dropout_model_seed,
            n_dropout_masks=self.n_dropout_masks,
            max_order=self.max_order,
            top_n=self.top_n,
            batch_size=self.batch_size,
            cpu_threads=self.cpu_threads,
        )

    def output(self):
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'uncertainty_models'
            / f'seeds_{self.seed_start}_{self.seed_end}'
            / f'order_{self.max_order}'
        )
        return {
            key: law.LocalFileTarget(path)
            for key, path in uncertainty_taylor_normalized_plot_paths(
                output_root,
                self.process,
                self.top_n,
            ).items()
        }

    def run(self):
        plot_uncertainty_model_taylor_process_normalized_to_max(
            process=self.process,
            output_dir=(
                WORKFLOW_ROOT
                / 'Taylor_coefficient_analysis'
                / 'uncertainty_models'
                / f'seeds_{self.seed_start}_{self.seed_end}'
                / f'order_{self.max_order}'
            ),
            top_n=self.top_n,
        )


class PlotNormalizedUncertaintyModelTaylorCoefficients(law.Task):
    """Plot normalized uncertainty Taylor results without recalculating coefficients."""

    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)
    dropout_model_seed = luigi.IntParameter(default=100)
    n_dropout_masks = luigi.IntParameter(default=100)
    max_order = luigi.IntParameter(default=2)
    top_n = luigi.IntParameter(default=10)
    batch_size = luigi.IntParameter(default=4096, significant=False)
    cpu_threads = luigi.IntParameter(default=8, significant=False)

    def requires(self):
        return {
            process: PlotNormalizedUncertaintyModelTaylorCoefficientsProcess(
                process=process,
                seed_start=self.seed_start,
                seed_end=self.seed_end,
                dropout_model_seed=self.dropout_model_seed,
                n_dropout_masks=self.n_dropout_masks,
                max_order=self.max_order,
                top_n=self.top_n,
                batch_size=self.batch_size,
                cpu_threads=self.cpu_threads,
            )
            for process in UNCERTAINTY_PROCESSES
        }

    def output(self):
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'uncertainty_models'
            / f'seeds_{self.seed_start}_{self.seed_end}'
            / f'order_{self.max_order}'
        )
        outputs = {
            f'{process}_{key}': law.LocalFileTarget(path)
            for process in UNCERTAINTY_PROCESSES
            for key, path in uncertainty_taylor_normalized_artifact_paths(
                output_root,
                process,
                self.top_n,
            ).items()
        }
        outputs['manifest'] = law.LocalFileTarget(
            output_root / 'normalized_to_max_manifest.json'
        )
        return outputs

    def run(self):
        write_uncertainty_taylor_normalized_manifest(
            output_dir=Path(self.output()['manifest'].path).parent,
            seed_start=self.seed_start,
            seed_end=self.seed_end,
            dropout_model_seed=self.dropout_model_seed,
            n_dropout_masks=self.n_dropout_masks,
            max_order=self.max_order,
            top_n=self.top_n,
            batch_size=self.batch_size,
            cpu_threads=self.cpu_threads,
        )


class PlotNormalizedUncertaintyModelTaylorCoefficientsSingleOrderProcess(
    law.Task
):
    """Plot one normalized uncertainty Taylor order for one process."""

    process = luigi.ChoiceParameter(choices=UNCERTAINTY_PROCESSES)
    taylor_order = luigi.IntParameter(default=1)
    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)
    dropout_model_seed = luigi.IntParameter(default=100)
    n_dropout_masks = luigi.IntParameter(default=100)
    max_order = luigi.IntParameter(default=1)
    top_n = luigi.IntParameter(default=10)
    batch_size = luigi.IntParameter(default=4096, significant=False)
    cpu_threads = luigi.IntParameter(default=8, significant=False)

    def requires(self):
        if self.taylor_order not in (1, 2):
            raise ValueError("taylor_order must be 1 or 2")
        if self.max_order < self.taylor_order:
            raise ValueError("max_order must be at least taylor_order")
        return CalculateUncertaintyModelTaylorCoefficientsProcess(
            process=self.process,
            seed_start=self.seed_start,
            seed_end=self.seed_end,
            dropout_model_seed=self.dropout_model_seed,
            n_dropout_masks=self.n_dropout_masks,
            max_order=self.max_order,
            top_n=self.top_n,
            batch_size=self.batch_size,
            cpu_threads=self.cpu_threads,
        )

    def output(self):
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'uncertainty_models'
            / f'seeds_{self.seed_start}_{self.seed_end}'
            / f'order_{self.max_order}'
        )
        return {
            key: law.LocalFileTarget(path)
            for key, path in uncertainty_taylor_normalized_single_order_plot_paths(
                output_root,
                self.process,
                self.top_n,
                self.taylor_order,
            ).items()
        }

    def run(self):
        plot_uncertainty_model_taylor_process_normalized_single_order(
            process=self.process,
            output_dir=(
                WORKFLOW_ROOT
                / 'Taylor_coefficient_analysis'
                / 'uncertainty_models'
                / f'seeds_{self.seed_start}_{self.seed_end}'
                / f'order_{self.max_order}'
            ),
            top_n=self.top_n,
            taylor_order=self.taylor_order,
        )


class PlotNormalizedUncertaintyModelTaylorCoefficientsSingleOrder(law.Task):
    """Plot normalized uncertainty Taylor coefficients for one order only."""

    taylor_order = luigi.IntParameter(default=1)
    seed_start = luigi.IntParameter(default=100)
    seed_end = luigi.IntParameter(default=199)
    dropout_model_seed = luigi.IntParameter(default=100)
    n_dropout_masks = luigi.IntParameter(default=100)
    max_order = luigi.IntParameter(default=1)
    top_n = luigi.IntParameter(default=10)
    batch_size = luigi.IntParameter(default=4096, significant=False)
    cpu_threads = luigi.IntParameter(default=8, significant=False)

    def requires(self):
        if self.taylor_order not in (1, 2):
            raise ValueError("taylor_order must be 1 or 2")
        if self.max_order < self.taylor_order:
            raise ValueError("max_order must be at least taylor_order")
        return {
            process: PlotNormalizedUncertaintyModelTaylorCoefficientsSingleOrderProcess(
                process=process,
                taylor_order=self.taylor_order,
                seed_start=self.seed_start,
                seed_end=self.seed_end,
                dropout_model_seed=self.dropout_model_seed,
                n_dropout_masks=self.n_dropout_masks,
                max_order=self.max_order,
                top_n=self.top_n,
                batch_size=self.batch_size,
                cpu_threads=self.cpu_threads,
            )
            for process in UNCERTAINTY_PROCESSES
        }

    def output(self):
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'uncertainty_models'
            / f'seeds_{self.seed_start}_{self.seed_end}'
            / f'order_{self.max_order}'
        )
        outputs = {
            f'{process}_{key}': law.LocalFileTarget(path)
            for process in UNCERTAINTY_PROCESSES
            for key, path in uncertainty_taylor_normalized_single_order_artifact_paths(
                output_root,
                process,
                self.top_n,
                self.taylor_order,
            ).items()
        }
        outputs['manifest'] = law.LocalFileTarget(
            output_root
            / f'normalized_to_max_order_{self.taylor_order}_manifest.json'
        )
        return outputs

    def run(self):
        write_uncertainty_taylor_normalized_single_order_manifest(
            output_dir=Path(self.output()['manifest'].path).parent,
            seed_start=self.seed_start,
            seed_end=self.seed_end,
            dropout_model_seed=self.dropout_model_seed,
            n_dropout_masks=self.n_dropout_masks,
            max_order=self.max_order,
            top_n=self.top_n,
            taylor_order=self.taylor_order,
            batch_size=self.batch_size,
            cpu_threads=self.cpu_threads,
        )


class PlotFirstOrderUncertaintyModelTaylorCoefficients(
    PlotUncertaintyModelTaylorCoefficients
):
    """Plot first-order uncertainty Taylor coefficients."""

    max_order = luigi.IntParameter(default=1)


class PlotFirstOrderNormalizedUncertaintyModelTaylorCoefficients(
    PlotNormalizedUncertaintyModelTaylorCoefficients
):
    """Plot first-order normalized uncertainty Taylor coefficients."""

    max_order = luigi.IntParameter(default=1)


class TaylorCoefficientComparison(law.Task):
    """Compare both Taylor methods for grouped and single W+jets DNNs."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    grouping = luigi.ChoiceParameter(
        default='njets',
        choices=ENRICHMENT_GROUPINGS,
    )
    reduced_weight_grouping = luigi.ChoiceParameter(
        default='tau_decaymode_2_alt',
        choices=ENRICHMENT_GROUPINGS,
    )
    max_order = luigi.IntParameter(default=2)
    top_n = luigi.IntParameter(default=10)
    batch_size = luigi.IntParameter(default=1024, significant=False)

    def requires(self):
        return {
            'grouped_dnn': TrainSqueezedModels(
                squeezing=self.squeezing
            ),
            'single_dnn': TrainSqueezedSingleDNNModels(
                squeezing=self.squeezing,
                reduced_weight_grouping=self.reduced_weight_grouping,
            ),
        }

    def output(self):
        output_dir = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_comparison'
            / squeezing_label(self.squeezing)
            / self.grouping
            / f'single_weights_{self.reduced_weight_grouping}'
            / f'order_{self.max_order}'
            / 'full_dataset'
            / f'top_{self.top_n}'
        )
        return {
            f'{model_type}_{artifact}': law.LocalFileTarget(
                output_dir
                / (
                        f'{model_type}_metadata.json'
                        if artifact == 'metadata'
                        else (
                            f'{model_type}_taylor_coefficients_'
                        f'{method}.{extension}'
                    )
                )
            )
            for model_type in ('grouped_dnn', 'single_dnn')
            for artifact, method, extension in (
                ('new_json', 'new_way', 'json'),
                ('new_png', 'new_way', 'png'),
                ('new_pdf', 'new_way', 'pdf'),
                ('notebook_json', 'notebook_way', 'json'),
                ('notebook_png', 'notebook_way', 'png'),
                ('notebook_pdf', 'notebook_way', 'pdf'),
                ('metadata', '', ''),
            )
        }

    def run(self):
        grouped_models_dir = Path(self.input()['grouped_dnn'].path).parent
        single_models_dir = Path(
            self.input()['single_dnn']['metadata'].path
        ).parent
        output_dir = Path(
            self.output()['grouped_dnn_metadata'].path
        ).parent
        run_taylor_coefficient_comparison(
            model_path=(
                grouped_models_dir
                / self.grouping
                / 'wjets'
                / 'fold_even'
            ),
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_var_path=(
                PROJECT_ROOT / 'configs' / 'training_variables.yaml'
            ),
            output_dir=output_dir,
            grouping=self.grouping,
            model_type='GroupedDNN',
            filename_prefix='grouped_dnn_',
            max_order=self.max_order,
            top_n=self.top_n,
            batch_size=self.batch_size,
        )
        run_taylor_coefficient_comparison(
            model_path=single_models_dir / 'wjets' / 'fold_even',
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_var_path=(
                PROJECT_ROOT / 'configs' / 'training_variables.yaml'
            ),
            output_dir=output_dir,
            grouping=self.reduced_weight_grouping,
            model_type='DNN',
            filename_prefix='single_dnn_',
            max_order=self.max_order,
            top_n=self.top_n,
            batch_size=self.batch_size,
        )


class ConvertFFModelsToONNX(law.Task):
    """Build and export combined FF models from squeezed-training outputs."""

    squeezing = luigi.OptionalFloatParameter(default=None)

    def requires(self):
        return TrainSqueezedModels(squeezing=self.squeezing)

    def output(self):
        output_dir = (
            WORKFLOW_ROOT
            / 'CombinedModels'
            / squeezing_label(self.squeezing)
        )
        outputs = {
            f"{process}_{grouping}_onnx": law.LocalFileTarget(
                output_dir
                / FF_MODEL_OUTPUT_NAMES[process]
                / grouping
                / 'onnx_model'
                / 'model.onnx'
            )
            for process in ('wjets', 'qcd', 'ttbar')
            for grouping in ENRICHMENT_GROUPINGS
        }
        outputs.update({
            f"{process}_{grouping}_torch": law.LocalFileTarget(
                output_dir
                / FF_MODEL_OUTPUT_NAMES[process]
                / grouping
                / 'torch_model'
                / 'model_weights.pth'
            )
            for process in ('wjets', 'qcd', 'ttbar')
            for grouping in ENRICHMENT_GROUPINGS
        })
        outputs['normalization_constants'] = law.LocalFileTarget(
            output_dir / 'normalization_constants.json'
        )
        return outputs

    def run(self):
        trained_models_dir = Path(self.input().path).parent
        output_dir = (
            WORKFLOW_ROOT
            / 'CombinedModels'
            / squeezing_label(self.squeezing)
        )
        ff_models_to_onnx(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            trained_models_dir=trained_models_dir,
            reduced_weight_dir=WORKFLOW_FEATURE_ROOT / 'reduced_dataset',
            output_dir=output_dir,
        )


class ConvertCorrectedFFModelsToONNX(law.Task):
    """Build and export DRSR-corrected combined FF models."""

    squeezing = luigi.FloatParameter(default=0.99)
    squeezing_loss_limit = luigi.FloatParameter(default=0.1)
    grouping = luigi.ChoiceParameter(default='njets', choices=('njets',))

    def requires(self):
        return {
            'combined_models': ConvertFFModelsToONNX(
                squeezing=self.squeezing,
            ),
            'drsr_models': TrainDRSRSqueezedModels(
                squeezing=self.squeezing,
                squeezing_loss_limit=self.squeezing_loss_limit,
            ),
        }

    def output(self):
        output_dir = (
            WORKFLOW_ROOT
            / 'CombinedModelsCorrected'
            / squeezing_label(self.squeezing)
        )
        output_dir = drsr_limit_subdir(output_dir, self.squeezing_loss_limit)
        outputs = {
            f"{process}_{self.grouping}_onnx": law.LocalFileTarget(
                output_dir
                / FF_MODEL_OUTPUT_NAMES[process]
                / self.grouping
                / 'onnx_model'
                / 'model.onnx'
            )
            for process in ('wjets', 'qcd', 'ttbar')
        }
        outputs.update({
            f"{process}_{self.grouping}_torch": law.LocalFileTarget(
                output_dir
                / FF_MODEL_OUTPUT_NAMES[process]
                / self.grouping
                / 'torch_model'
                / 'model_weights.pth'
            )
            for process in ('wjets', 'qcd', 'ttbar')
        })
        outputs['metadata'] = law.LocalFileTarget(
            output_dir / 'metadata.json'
        )
        return outputs

    def run(self):
        combined_model_dir = Path(
            self.input()['combined_models']['normalization_constants'].path
        ).parent
        drsr_model_dir = drsr_squeezed_model_dir(
            self.squeezing,
            self.squeezing_loss_limit,
        )
        output_dir = Path(self.output()['metadata'].path).parent
        corrected_ff_models_to_onnx(
            combined_model_dir=combined_model_dir,
            drsr_model_dir=drsr_model_dir,
            output_dir=output_dir,
            grouping=self.grouping,
        )


class CalculateFakeFactors(law.Task):
    """Evaluate combined FF models and add their outputs as lazy features."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    batch_size = luigi.IntParameter(default=65536, significant=False)
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )

    def requires(self):
        return ConvertFFModelsToONNX(squeezing=self.squeezing)

    def output(self):
        output_dir = (
            WORKFLOW_FEATURE_ROOT
            / 'fake_factors'
            / squeezing_label(self.squeezing)
        )
        return {
            'features': law.LocalFileTarget(
                output_dir / 'fake_factors.feather'
            ),
            'schema': law.LocalFileTarget(
                output_dir / '.schema_v4_squeezing_feature_names'
            ),
        }

    def run(self):
        combined_models_dir = Path(
            self.input()['normalization_constants'].path
        ).parent
        calculate_and_store_fake_factors(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_variables_path=(
                PROJECT_ROOT / 'configs' / 'training_variables.yaml'
            ),
            combined_models_dir=combined_models_dir,
            feature_store_path=self.output()['features'].path,
            feature_registry_path=(
                WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
            ),
            process_fractions_path=self.process_fractions_path,
            batch_size=self.batch_size,
            feature_suffix=squeezing_feature_suffix(self.squeezing),
        )
        Path(self.output()['schema'].path).write_text(
            "nine process FF features plus three combined FFs in SR/AR\n"
        )


class CalculateExtrapolationCorrectedFakeFactors(law.Task):
    """Evaluate DRSR/extrapolation-corrected FF models for njets only."""

    squeezing = luigi.FloatParameter(default=0.99)
    squeezing_loss_limit = luigi.FloatParameter(default=0.1)
    batch_size = luigi.IntParameter(default=65536, significant=False)
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )

    def requires(self):
        return ConvertCorrectedFFModelsToONNX(
            squeezing=self.squeezing,
            squeezing_loss_limit=self.squeezing_loss_limit,
            grouping='njets',
        )

    def output(self):
        output_dir = (
            WORKFLOW_FEATURE_ROOT
            / 'fake_factors_extrapolation_corrected'
            / squeezing_label(self.squeezing)
        )
        output_dir = drsr_limit_subdir(output_dir, self.squeezing_loss_limit)
        return {
            'features': law.LocalFileTarget(
                output_dir / 'fake_factors_extrapolation_corrected.feather'
            ),
            'schema': law.LocalFileTarget(
                output_dir / '.schema_v1_njets_squeezing_feature_names'
            ),
        }

    def run(self):
        combined_models_dir = Path(
            self.input()['metadata'].path
        ).parent
        calculate_and_store_fake_factors(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_variables_path=(
                PROJECT_ROOT / 'configs' / 'training_variables.yaml'
            ),
            combined_models_dir=combined_models_dir,
            feature_store_path=self.output()['features'].path,
            feature_registry_path=(
                WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
            ),
            process_fractions_path=self.process_fractions_path,
            batch_size=self.batch_size,
            feature_suffix=squeezing_feature_suffix(self.squeezing),
            groupings=('njets',),
        )
        Path(self.output()['schema'].path).write_text(
            "DRSR/extrapolation-corrected njets process FF features plus "
            "combined ff_dnn_njets in SR/AR\n"
        )


def drsr_squeezed_model_dir(
    squeezing,
    squeezing_loss_limit=0.1,
):
    return (
        WORKFLOW_ROOT
        / 'DRSR_models_squeezed'
        / squeezing_label(squeezing)
        / drsr_loss_limit_label(squeezing_loss_limit)
    )


def drsr_limit_subdir(base_dir, squeezing_loss_limit):
    return Path(base_dir) / drsr_loss_limit_label(squeezing_loss_limit)


class TrainDRSRSqueezedModels(law.Task):
    """Train DR-vs-SR models using grouped-DNN FF-weighted backgrounds."""

    squeezing = luigi.FloatParameter(default=0.99)
    squeezing_loss_limit = luigi.FloatParameter(default=0.1)

    def requires(self):
        return {
            'combined_models': ConvertFFModelsToONNX(
                squeezing=self.squeezing,
            ),
            'qcd_extrapolation': TrainEnrichmentQCDExtrapolation(),
        }

    def output(self):
        output_dir = drsr_squeezed_model_dir(
            self.squeezing,
            self.squeezing_loss_limit,
        )
        outputs = {
            f'{process}_torch': law.LocalFileTarget(
                output_dir / process / 'model_weights.pth'
            )
            for process in ('wjets', 'qcd', 'ttbar')
        }
        outputs['metadata'] = law.LocalFileTarget(
            output_dir / 'metadata.json'
        )
        return outputs

    def run(self):
        combined_model_dir = Path(
            self.input()['combined_models']['normalization_constants'].path
        ).parent
        qcd_extrapolation_feature_path = (
            WORKFLOW_FEATURE_ROOT
            / 'qcd_extrapolation'
            / 'qcd_extrapolation_weights_njets.feather'
        )
        output_dir = drsr_squeezed_model_dir(
            self.squeezing,
            self.squeezing_loss_limit,
        )
        train_drsr_squeezed_models(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_var_path=(
                PROJECT_ROOT / 'configs' / 'training_variables.yaml'
            ),
            nn_config_path=PROJECT_ROOT / 'configs' / 'DNN.yaml',
            combined_model_dir=combined_model_dir,
            qcd_extrapolation_feature_path=qcd_extrapolation_feature_path,
            output_dir=output_dir.parent,
            squeezing_loss_limit=self.squeezing_loss_limit,
        )


class CalculateDRSRCorrectionFactors(law.Task):
    """Evaluate DRSR correction factors C=NN/(1-NN) on all AR rows."""

    squeezing = luigi.FloatParameter(default=0.99)
    squeezing_loss_limit = luigi.FloatParameter(default=0.1)
    batch_size = luigi.IntParameter(default=65536, significant=False)

    def requires(self):
        return TrainDRSRSqueezedModels(
            squeezing=self.squeezing,
            squeezing_loss_limit=self.squeezing_loss_limit,
        )

    def output(self):
        output_dir = (
            WORKFLOW_FEATURE_ROOT
            / 'drsr_corrections'
            / squeezing_label(self.squeezing)
        )
        output_dir = drsr_limit_subdir(output_dir, self.squeezing_loss_limit)
        return {
            'features': law.LocalFileTarget(
                output_dir / 'drsr_corrections.feather'
            ),
            'schema': law.LocalFileTarget(
                output_dir / '.schema_v2_all_ar'
            ),
        }

    def run(self):
        drsr_model_dir = drsr_squeezed_model_dir(
            self.squeezing,
            self.squeezing_loss_limit,
        )
        feature_path = calculate_and_store_drsr_correction_factors(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_variables_path=(
                PROJECT_ROOT / 'configs' / 'training_variables.yaml'
            ),
            drsr_model_dir=drsr_model_dir,
            feature_store_path=self.output()['features'].path,
            feature_registry_path=(
                WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
            ),
            squeezing=self.squeezing,
            batch_size=self.batch_size,
        )
        correction_columns = [
            drsr_correction_name(process, squeezing=self.squeezing)
            for process in ('wjets', 'qcd', 'ttbar')
        ]
        Path(self.output()['schema'].path).write_text(
            ", ".join(correction_columns)
            + " calculated for all AR rows\n"
        )
        print("DRSR CORRECTION OUTPUT:", feature_path)


class CalculateTrainingFractionFakeFactors(law.Task):
    """Combine process FFs with NN-trained process fractions in all AR rows."""

    grouping = luigi.ChoiceParameter(
        default='njets',
        choices=GROUPING_NAMES,
    )
    squeezing = luigi.OptionalFloatParameter(default=0.99)
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )

    def requires(self):
        return {
            'fractions': PlotFractions(
                process_fractions_path=self.process_fractions_path,
            ),
            'fake_factors': CalculateFakeFactors(
                squeezing=self.squeezing,
                process_fractions_path=self.process_fractions_path,
            ),
        }

    def output(self):
        feature_name = training_fraction_fake_factor_name(
            grouping=self.grouping,
            squeezing=self.squeezing,
        )
        output_dir = (
            WORKFLOW_FEATURE_ROOT
            / 'training_fraction_fake_factors'
            / squeezing_label(self.squeezing)
        )
        return {
            'features': law.LocalFileTarget(
                output_dir / f'{feature_name}.feather'
            ),
            'schema': law.LocalFileTarget(
                output_dir / f'.schema_v2_all_ar_{feature_name}'
            ),
        }

    def complete(self):
        feature_name = training_fraction_fake_factor_name(
            grouping=self.grouping,
            squeezing=self.squeezing,
        )
        return (
            super().complete()
            and ar_feature_file_is_complete(
                self.output()['features'].path,
                feature_name,
            )
        )

    def run(self):
        feature_path = calculate_and_store_training_fraction_fake_factors(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            fraction_feature_path=self.input()['fractions']['features'].path,
            fake_factor_feature_path=(
                self.input()['fake_factors']['features'].path
            ),
            feature_store_path=self.output()['features'].path,
            feature_registry_path=(
                WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
            ),
            grouping=self.grouping,
            squeezing=self.squeezing,
        )
        Path(self.output()['schema'].path).write_text(
            "training-fraction combined DNN fake factor in AR, "
            "keyed by dataframe row index\n"
        )
        print("TRAINING FRACTION FAKE FACTOR OUTPUT:", feature_path)


class CalculateCorrectedTrainingFractionFakeFactors(law.Task):
    """Apply non-closure corrections to the MLF-combined fake factor."""

    grouping = luigi.ChoiceParameter(
        default='njets',
        choices=GROUPING_NAMES,
    )
    squeezing = luigi.OptionalFloatParameter(default=0.99)
    correction_set_root = law.Parameter(
        default="/work/mmoser/TauFakeFactors/workdirs"
    )
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )

    def requires(self):
        return {
            'mlf': CalculateTrainingFractionFakeFactors(
                grouping=self.grouping,
                squeezing=self.squeezing,
                process_fractions_path=self.process_fractions_path,
            ),
            'fractions': PlotFractions(
                process_fractions_path=self.process_fractions_path,
            ),
            'fake_factors': CalculateFakeFactors(
                squeezing=self.squeezing,
                process_fractions_path=self.process_fractions_path,
            ),
        }

    def output(self):
        feature_name = corrected_training_fraction_fake_factor_name(
            grouping=self.grouping,
            squeezing=self.squeezing,
        )
        output_dir = (
            WORKFLOW_FEATURE_ROOT
            / 'training_fraction_fake_factors_corrected'
            / squeezing_label(self.squeezing)
        )
        return {
            'features': law.LocalFileTarget(
                output_dir / f'{feature_name}.feather'
            ),
            'schema': law.LocalFileTarget(
                output_dir / f'.schema_v2_all_ar_{feature_name}'
            ),
        }

    def complete(self):
        feature_name = corrected_training_fraction_fake_factor_name(
            grouping=self.grouping,
            squeezing=self.squeezing,
        )
        return (
            super().complete()
            and ar_feature_file_is_complete(
                self.output()['features'].path,
                feature_name,
            )
        )

    def run(self):
        feature_path = (
            calculate_and_store_corrected_training_fraction_fake_factors(
                data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
                masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
                fraction_feature_path=(
                    self.input()['fractions']['features'].path
                ),
                fake_factor_feature_path=(
                    self.input()['fake_factors']['features'].path
                ),
                correction_set_root=self.correction_set_root,
                feature_store_path=self.output()['features'].path,
                feature_registry_path=(
                    WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
                ),
                grouping=self.grouping,
                squeezing=self.squeezing,
                squeezing_loss_limit=self.squeezing_loss_limit,
            )
        )
        Path(self.output()['schema'].path).write_text(
            "non-closure-corrected MLF DNN fake factor in AR, "
            "keyed by dataframe row index\n"
        )
        print("CORRECTED TRAINING FRACTION FAKE FACTOR OUTPUT:", feature_path)


class CalculateExtrapolationCorrectedTrainingFractionFakeFactors(law.Task):
    """
    Combine ML fractions with DRSR/extrapolation-corrected process FF models
    and TauFakeFactors non-closure corrections from the extrapolation workdir.
    """

    grouping = luigi.ChoiceParameter(default='njets', choices=('njets',))
    squeezing = luigi.FloatParameter(default=0.99)
    squeezing_loss_limit = luigi.FloatParameter(default=0.1)
    correction_set_root = law.Parameter(
        default="/work/mmoser/TauFakeFactors/workdirs_with_extrapolation_correction"
    )
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )

    def requires(self):
        return {
            'fractions': PlotFractions(
                process_fractions_path=self.process_fractions_path,
            ),
            'fake_factors': CalculateExtrapolationCorrectedFakeFactors(
                squeezing=self.squeezing,
                squeezing_loss_limit=self.squeezing_loss_limit,
                process_fractions_path=self.process_fractions_path,
            ),
        }

    def output(self):
        feature_name = extrapolation_corrected_training_fraction_fake_factor_name(
            grouping=self.grouping,
            squeezing=self.squeezing,
        )
        output_dir = (
            WORKFLOW_FEATURE_ROOT
            / 'training_fraction_fake_factors_extrapolation_corrected'
            / squeezing_label(self.squeezing)
        )
        output_dir = drsr_limit_subdir(output_dir, self.squeezing_loss_limit)
        return {
            'features': law.LocalFileTarget(
                output_dir / f'{feature_name}.feather'
            ),
            'schema': law.LocalFileTarget(
                output_dir / f'.schema_v1_all_ar_{feature_name}'
            ),
        }

    def complete(self):
        feature_name = extrapolation_corrected_training_fraction_fake_factor_name(
            grouping=self.grouping,
            squeezing=self.squeezing,
        )
        return (
            super().complete()
            and ar_feature_file_is_complete(
                self.output()['features'].path,
                feature_name,
            )
        )

    def run(self):
        feature_path = (
            calculate_and_store_extrapolation_corrected_training_fraction_fake_factors(
                data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
                masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
                fraction_feature_path=self.input()['fractions']['features'].path,
                fake_factor_feature_path=(
                    self.input()['fake_factors']['features'].path
                ),
                correction_set_root=self.correction_set_root,
                feature_store_path=self.output()['features'].path,
                feature_registry_path=(
                    WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
                ),
                grouping=self.grouping,
                squeezing=self.squeezing,
            )
        )
        feature_name = extrapolation_corrected_training_fraction_fake_factor_name(
            grouping=self.grouping,
            squeezing=self.squeezing,
        )
        Path(self.output()['schema'].path).write_text(
            f"{feature_name} calculated for all AR rows using "
            "DRSR/extrapolation-corrected process models and "
            "TauFakeFactors non-closure corrections from "
            f"{self.correction_set_root}\n"
        )
        print(
            "EXTRAPOLATION-CORRECTED TRAINING FRACTION FAKE FACTOR OUTPUT:",
            feature_path,
        )


class CalculateDRSRCorrectedTrainingFractionFakeFactors(law.Task):
    """Apply DRSR process corrections to the MLF-combined fake factor."""

    grouping = luigi.ChoiceParameter(
        default='njets',
        choices=GROUPING_NAMES,
    )
    squeezing = luigi.FloatParameter(default=0.99)
    squeezing_loss_limit = luigi.FloatParameter(default=0.1)
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )

    def requires(self):
        return {
            'fractions': PlotFractions(
                process_fractions_path=self.process_fractions_path,
            ),
            'fake_factors': CalculateFakeFactors(
                squeezing=self.squeezing,
                process_fractions_path=self.process_fractions_path,
            ),
            'drsr_corrections': CalculateDRSRCorrectionFactors(
                squeezing=self.squeezing,
                squeezing_loss_limit=self.squeezing_loss_limit,
            ),
        }

    def output(self):
        feature_name = drsr_corrected_training_fraction_fake_factor_name(
            grouping=self.grouping,
            squeezing=self.squeezing,
        )
        output_dir = (
            WORKFLOW_FEATURE_ROOT
            / 'training_fraction_fake_factors_drsr_corrected'
            / squeezing_label(self.squeezing)
        )
        output_dir = drsr_limit_subdir(output_dir, self.squeezing_loss_limit)
        return {
            'features': law.LocalFileTarget(
                output_dir / f'{feature_name}.feather'
            ),
            'schema': law.LocalFileTarget(
                output_dir / f'.schema_v2_all_ar_{feature_name}'
            ),
        }

    def complete(self):
        feature_name = drsr_corrected_training_fraction_fake_factor_name(
            grouping=self.grouping,
            squeezing=self.squeezing,
        )
        return (
            super().complete()
            and ar_feature_file_is_complete(
                self.output()['features'].path,
                feature_name,
            )
        )

    def run(self):
        feature_path = (
            calculate_and_store_drsr_corrected_training_fraction_fake_factors(
                data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
                masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
                fraction_feature_path=self.input()['fractions']['features'].path,
                fake_factor_feature_path=(
                    self.input()['fake_factors']['features'].path
                ),
                drsr_correction_feature_path=(
                    self.input()['drsr_corrections']['features'].path
                ),
                feature_store_path=self.output()['features'].path,
                feature_registry_path=(
                    WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
                ),
                grouping=self.grouping,
                squeezing=self.squeezing,
            )
        )
        feature_name = drsr_corrected_training_fraction_fake_factor_name(
            grouping=self.grouping,
            squeezing=self.squeezing,
        )
        Path(self.output()['schema'].path).write_text(
            f"{feature_name} calculated for all AR rows\n"
        )
        print("DRSR-CORRECTED TRAINING FRACTION FAKE FACTOR OUTPUT:", feature_path)


class PlotClosure(law.Task):
    """Plot inclusive and njets-split closures for corrected MLF FFs."""

    grouping = luigi.ChoiceParameter(
        default='njets',
        choices=('njets',),
    )
    squeezing = luigi.OptionalFloatParameter(default=0.99)
    variable_set = luigi.Parameter(default='variables_set_large')
    correction_set_root = law.Parameter(
        default="/work/mmoser/TauFakeFactors/workdirs"
    )
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )
    classic_corrections_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/FF_corrections_et.json.gz"
        )
    )

    def requires(self):
        return {
            'mlf_corrected': CalculateCorrectedTrainingFractionFakeFactors(
                grouping=self.grouping,
                squeezing=self.squeezing,
                correction_set_root=self.correction_set_root,
                process_fractions_path=self.process_fractions_path,
            ),
            'classic': CalculateClassicFakeFactors(
                fake_factors_path=self.process_fractions_path,
                corrections_path=self.classic_corrections_path,
            ),
        }

    def output(self):
        return law.LocalFileTarget(
            WORKFLOW_ROOT
            / 'plots'
            / 'training_fraction_fake_factors'
            / squeezing_label(self.squeezing)
            / self.grouping
            / self.variable_set
            / 'manifest_closures_v1.json'
        )

    def run(self):
        create_mlf_closure_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            mlf_feature_path=(
                self.input()['mlf_corrected']['features'].path
            ),
            classic_feature_path=self.input()['classic']['features'].path,
            plotting_config_path=PROJECT_ROOT / 'configs' / 'plotting.yaml',
            labels_path=PROJECT_ROOT / 'configs' / 'labels.yaml',
            output_dir=Path(self.output().path).parent,
            manifest_path=self.output().path,
            variable_set=self.variable_set,
            mlf_column=corrected_training_fraction_fake_factor_name(
                grouping=self.grouping,
                squeezing=self.squeezing,
            ),
        )


class PlotExtrapolationCorrectedClosure(law.Task):
    """Plot closures for extrapolation-corrected MLF fake factors."""

    grouping = luigi.ChoiceParameter(default='njets', choices=('njets',))
    squeezing = luigi.FloatParameter(default=0.99)
    squeezing_loss_limit = luigi.FloatParameter(default=0.1)
    variable_set = luigi.Parameter(default='variables_set_large')
    correction_set_root = law.Parameter(
        default="/work/mmoser/TauFakeFactors/workdirs_with_extrapolation_correction"
    )
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )
    classic_corrections_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/FF_corrections_et.json.gz"
        )
    )

    def requires(self):
        return {
            'mlf_extrapolation_corrected': (
                CalculateExtrapolationCorrectedTrainingFractionFakeFactors(
                    grouping=self.grouping,
                    squeezing=self.squeezing,
                    squeezing_loss_limit=self.squeezing_loss_limit,
                    correction_set_root=self.correction_set_root,
                    process_fractions_path=self.process_fractions_path,
                )
            ),
            'classic': CalculateClassicFakeFactors(
                fake_factors_path=self.process_fractions_path,
                corrections_path=self.classic_corrections_path,
            ),
        }

    def output(self):
        output_dir = (
            WORKFLOW_ROOT
            / 'plots'
            / 'training_fraction_fake_factors_extrapolation_corrected'
            / squeezing_label(self.squeezing)
        )
        output_dir = drsr_limit_subdir(output_dir, self.squeezing_loss_limit)
        return law.LocalFileTarget(
            output_dir
            / self.grouping
            / self.variable_set
            / 'manifest_closures_v1.json'
        )

    def run(self):
        create_extrapolation_corrected_mlf_closure_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            mlf_feature_path=(
                self.input()['mlf_extrapolation_corrected']['features'].path
            ),
            classic_feature_path=self.input()['classic']['features'].path,
            plotting_config_path=PROJECT_ROOT / 'configs' / 'plotting.yaml',
            labels_path=PROJECT_ROOT / 'configs' / 'labels.yaml',
            output_dir=Path(self.output().path).parent,
            manifest_path=self.output().path,
            variable_set=self.variable_set,
            mlf_column=extrapolation_corrected_training_fraction_fake_factor_name(
                grouping=self.grouping,
                squeezing=self.squeezing,
            ),
        )


class CalculateClassicFakeFactors(law.Task):
    """Calculate classic fake factors and store them as lazy features."""

    fake_factors_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )
    corrections_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/FF_corrections_et.json.gz"
        )
    )

    def requires(self):
        return BuildDataset()

    def output(self):
        output_dir = WORKFLOW_FEATURE_ROOT / 'classic_fake_factors'
        return {
            'features': law.LocalFileTarget(
                output_dir / 'classic_fake_factors.feather'
            ),
            'schema': law.LocalFileTarget(
                output_dir / '.schema_v1_sr_ar'
            ),
        }

    def run(self):
        calculate_and_store_classic_fake_factors(
            data_path=self.input().path,
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            feature_store_path=self.output()['features'].path,
            feature_registry_path=(
                WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
            ),
            fake_factors_path=self.fake_factors_path,
            corrections_path=self.corrections_path,
        )
        Path(self.output()['schema'].path).write_text(
            "ff_classic calculated in SR and AR, keyed by dataframe row index\n"
        )


class CalculateCorrectedFakeFactors(law.Task):
    """Apply correctionlib corrections to DNN fake factors in AR."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    correction_set_root = law.Parameter(
        default="/work/mmoser/TauFakeFactors/workdirs"
    )
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )

    def requires(self):
        return {
            'data': BuildDataset(),
            'dnn': CalculateFakeFactors(
                squeezing=self.squeezing,
                process_fractions_path=self.process_fractions_path,
            ),
        }

    def output(self):
        output_dir = (
            WORKFLOW_FEATURE_ROOT
            / 'fake_factors_corrected'
            / squeezing_label(self.squeezing)
        )
        return {
            'features': law.LocalFileTarget(
                output_dir / 'fake_factors_corrected.feather'
            ),
            'schema': law.LocalFileTarget(
                output_dir / '.schema_v2_three_groupings_ar'
            ),
        }

    def run(self):
        calculate_and_store_corrected_fake_factors(
            data_path=self.input()['data'].path,
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            fake_factor_feature_path=(
                self.input()['dnn']['features'].path
            ),
            correction_set_root=self.correction_set_root,
            feature_store_path=self.output()['features'].path,
            feature_registry_path=(
                WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
            ),
            squeezing=self.squeezing,
        )
        Path(self.output()['schema'].path).write_text(
            "three corrected combined DNN fake factors in AR, "
            "keyed by dataframe row index\n"
        )


class PlotCorrectedFakeFactors(law.Task):
    """Mirror the FF closure workflow using corrected DNN fake factors."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    variable_set = luigi.Parameter(default='variables_set_small')
    correction_set_root = law.Parameter(
        default="/work/mmoser/TauFakeFactors/workdirs"
    )
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )
    classic_corrections_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/FF_corrections_et.json.gz"
        )
    )

    def requires(self):
        return {
            'corrected': CalculateCorrectedFakeFactors(
                squeezing=self.squeezing,
                correction_set_root=self.correction_set_root,
                process_fractions_path=self.process_fractions_path,
            ),
            'classic': CalculateClassicFakeFactors(
                fake_factors_path=self.process_fractions_path,
                corrections_path=self.classic_corrections_path,
            ),
        }

    def output(self):
        return law.LocalFileTarget(
            WORKFLOW_ROOT
            / 'plots'
            / 'fake_factors_corrected'
            / squeezing_label(self.squeezing)
            / self.variable_set
            / 'manifest_closures_v1.json'
        )

    def run(self):
        create_corrected_fake_factor_closure_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            corrected_feature_path=(
                self.input()['corrected']['features'].path
            ),
            classic_feature_path=self.input()['classic']['features'].path,
            plotting_config_path=PROJECT_ROOT / 'configs' / 'plotting.yaml',
            labels_path=PROJECT_ROOT / 'configs' / 'labels.yaml',
            output_dir=Path(self.output().path).parent,
            manifest_path=self.output().path,
            variable_set=self.variable_set,
            feature_suffix=squeezing_feature_suffix(self.squeezing),
        )


class PlotFakeFactors(law.Task):
    """Create closure and FF-distribution plots from workflow features."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    variable_set = luigi.Parameter(default='variables_set_small')
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )
    classic_corrections_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/FF_corrections_et.json.gz"
        )
    )

    def requires(self):
        return {
            'dnn': CalculateFakeFactors(
                squeezing=self.squeezing,
                process_fractions_path=self.process_fractions_path,
            ),
            'classic': CalculateClassicFakeFactors(
                fake_factors_path=self.process_fractions_path,
                corrections_path=self.classic_corrections_path,
            ),
            'reduced': ReducedDataset(),
        }

    def output(self):
        return law.LocalFileTarget(
            WORKFLOW_ROOT
            / 'plots'
            / 'fake_factors'
            / squeezing_label(self.squeezing)
            / self.variable_set
            / 'manifest_subsets_v8.json'
        )

    def run(self):
        create_fake_factor_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            feature_paths=(
                self.input()['dnn']['features'].path,
                self.input()['classic']['features'].path,
            ),
            reduced_weight_paths=(
                *(
                    Path(self.input()['reduced'][process].path)
                    / f'reduced_weight_{grouping}.feather'
                    for process in ('wjets', 'qcd')
                    for grouping in ENRICHMENT_GROUPINGS
                ),
            ),
            plotting_config_path=PROJECT_ROOT / 'configs' / 'plotting.yaml',
            labels_path=PROJECT_ROOT / 'configs' / 'labels.yaml',
            output_dir=Path(self.output().path).parent,
            manifest_path=self.output().path,
            variable_set=self.variable_set,
            feature_suffix=squeezing_feature_suffix(self.squeezing),
        )


class PlotFakeFactorDistributionsOppositeGrouping(law.Task):
    """Create inclusive grouped-DNN FF distributions in the opposite grouping."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )

    def requires(self):
        return CalculateFakeFactors(
            squeezing=self.squeezing,
            process_fractions_path=self.process_fractions_path,
        )

    def output(self):
        return law.LocalFileTarget(
            WORKFLOW_ROOT
            / 'plots'
            / 'fake_factor_distributions_opposite_grouping'
            / squeezing_label(self.squeezing)
            / 'manifest_grouped_dnn_v1.json'
        )

    def run(self):
        create_fake_factor_opposite_grouping_distribution_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            feature_path=self.input()['features'].path,
            output_dir=Path(self.output().path).parent,
            manifest_path=self.output().path,
            feature_suffix=squeezing_feature_suffix(self.squeezing),
        )


class PlotFakeFactorDistributionsAllSplits(
    PlotFakeFactorDistributionsOppositeGrouping
):
    """Backward-compatible task name for opposite-grouping distributions."""


class PlotDRSRProcessFakeFactorDistributions(law.Task):
    """Plot process FF distributions before and after DRSR correction."""

    process = luigi.ChoiceParameter(
        default='wjets',
        choices=('wjets', 'qcd', 'ttbar'),
    )
    squeezing = luigi.FloatParameter(default=0.99)
    squeezing_loss_limit = luigi.FloatParameter(default=0.1)
    value_min = luigi.FloatParameter(default=0.0)
    value_max = luigi.FloatParameter(default=10.0)
    n_bins = luigi.IntParameter(default=80)
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )

    def requires(self):
        return {
            'fake_factors': CalculateFakeFactors(
                squeezing=self.squeezing,
                process_fractions_path=self.process_fractions_path,
            ),
            'drsr_corrections': CalculateDRSRCorrectionFactors(
                squeezing=self.squeezing,
                squeezing_loss_limit=self.squeezing_loss_limit,
            ),
        }

    def output(self):
        output_dir = (
            WORKFLOW_ROOT
            / 'plots'
            / 'drsr_process_fake_factor_distributions'
            / squeezing_label(self.squeezing)
        )
        output_dir = drsr_limit_subdir(output_dir, self.squeezing_loss_limit)
        return law.LocalFileTarget(
            output_dir
            / self.process
            / f'{self.value_min:g}_{self.value_max:g}'
            / 'manifest_njets_v3_split_axis_legend_bins.json'
        )

    def run(self):
        create_drsr_process_fake_factor_distribution_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            fake_factor_feature_path=(
                self.input()['fake_factors']['features'].path
            ),
            drsr_correction_feature_path=(
                self.input()['drsr_corrections']['features'].path
            ),
            output_dir=Path(self.output().path).parent,
            manifest_path=self.output().path,
            process=self.process,
            squeezing=self.squeezing,
            value_min=self.value_min,
            value_max=self.value_max,
            n_bins=self.n_bins,
        )


class PlotDRSRCorrectionDistributions(law.Task):
    """Plot DRSR correction-factor distributions without FF distributions."""

    process = luigi.ChoiceParameter(
        default='all',
        choices=('all', 'wjets', 'qcd', 'ttbar'),
    )
    squeezing = luigi.FloatParameter(default=0.99)
    squeezing_loss_limit = luigi.FloatParameter(default=0.1)
    value_min = luigi.FloatParameter(default=0.0)
    value_max = luigi.FloatParameter(default=2.0)
    n_bins = luigi.IntParameter(default=100)

    def requires(self):
        return CalculateDRSRCorrectionFactors(
            squeezing=self.squeezing,
            squeezing_loss_limit=self.squeezing_loss_limit,
        )

    def output(self):
        output_dir = (
            WORKFLOW_ROOT
            / 'plots'
            / 'drsr_correction_distributions'
            / squeezing_label(self.squeezing)
        )
        output_dir = drsr_limit_subdir(output_dir, self.squeezing_loss_limit)
        return law.LocalFileTarget(
            output_dir
            / self.process
            / f'{self.value_min:g}_{self.value_max:g}'
            / f'manifest_inclusive_v2_linear_{self.n_bins}_bins.json'
        )

    def run(self):
        create_drsr_correction_distribution_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            drsr_correction_feature_path=self.input()['features'].path,
            output_dir=Path(self.output().path).parent,
            manifest_path=self.output().path,
            process=self.process,
            squeezing=self.squeezing,
            value_min=self.value_min,
            value_max=self.value_max,
            n_bins=self.n_bins,
        )


class PlotHighFakeFactorClosures(law.Task):
    """Plot model-only closure predictions for events with FF greater than one."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    variable_set = luigi.Parameter(default='variables_set_small')
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )

    def requires(self):
        return {
            'dnn': CalculateFakeFactors(
                squeezing=self.squeezing,
                process_fractions_path=self.process_fractions_path,
            ),
            'reduced': ReducedDataset(),
        }

    def output(self):
        return law.LocalFileTarget(
            WORKFLOW_ROOT
            / 'plots'
            / 'fake_factors'
            / squeezing_label(self.squeezing)
            / self.variable_set
            / 'high_ff_gt_1'
            / 'manifest_closures_v1.json'
        )

    def run(self):
        create_high_ff_closure_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            feature_path=self.input()['dnn']['features'].path,
            reduced_weight_paths=(
                *(
                    Path(self.input()['reduced'][process].path)
                    / f'reduced_weight_{grouping}.feather'
                    for process in ('wjets', 'qcd')
                    for grouping in ENRICHMENT_GROUPINGS
                ),
            ),
            plotting_config_path=PROJECT_ROOT / 'configs' / 'plotting.yaml',
            labels_path=PROJECT_ROOT / 'configs' / 'labels.yaml',
            output_dir=Path(self.output().path).parent,
            manifest_path=self.output().path,
            variable_set=self.variable_set,
            feature_suffix=squeezing_feature_suffix(self.squeezing),
        )


class PlotHighFakeFactorDistributions(law.Task):
    """Plot grouped AR and AR-like FF distributions between 1 and 100."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    value_min = luigi.FloatParameter(default=1.0)
    value_max = luigi.FloatParameter(default=100.0)
    n_bins = luigi.IntParameter(default=90)
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )

    def requires(self):
        return CalculateFakeFactors(
            squeezing=self.squeezing,
            process_fractions_path=self.process_fractions_path,
        )

    def output(self):
        return law.LocalFileTarget(
            WORKFLOW_ROOT
            / 'plots'
            / 'fake_factors_high_range'
            / squeezing_label(self.squeezing)
            / f'{self.value_min:g}_{self.value_max:g}'
            / 'manifest_groupings_v6.json'
        )

    def run(self):
        create_high_fake_factor_distribution_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            feature_path=self.input()['features'].path,
            output_dir=Path(self.output().path).parent,
            manifest_path=self.output().path,
            value_min=self.value_min,
            value_max=self.value_max,
            n_bins=self.n_bins,
            feature_suffix=squeezing_feature_suffix(self.squeezing),
        )


class TrainSqueezedSingleDNNModels(law.Task):
    """Train one ordinary two-fold DNN per process without grouping."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    reduced_weight_grouping = luigi.ChoiceParameter(
        default='tau_decaymode_2_alt',
        choices=ENRICHMENT_GROUPINGS,
    )

    def requires(self):
        return ReducedDataset()

    def output(self):
        output_dir = (
            WORKFLOW_ROOT
            / 'Training_results_squeezed_single_dnn'
            / self.reduced_weight_grouping
            / squeezing_label(self.squeezing)
        )
        return {
            f'{process}_{fold}': law.LocalFileTarget(
                output_dir / process / fold / 'model_weights.pth'
            )
            for process in ('wjets', 'qcd', 'ttbar')
            for fold in ('fold_even', 'fold_odd')
        } | {
            'metadata': law.LocalFileTarget(output_dir / 'metadata.json')
        }

    def run(self):
        train_squeezed_single_dnn_models(
            squeezing=self.squeezing,
            reduced_weight_grouping=self.reduced_weight_grouping,
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_var_path=PROJECT_ROOT / 'configs' / 'training_variables.yaml',
            nn_config_path=PROJECT_ROOT / 'configs' / 'DNN.yaml',
            checkpoint_dir=(
                WORKFLOW_ROOT / 'Training_results_squeezed_single_dnn'
            ),
            reduced_weight_dir=WORKFLOW_FEATURE_ROOT / 'reduced_dataset',
        )


class CalculateSingleDNNTaylorCoefficients(law.Task):
    """Calculate Taylor coefficients for all ordinary DNN process models."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    reduced_weight_grouping = luigi.ChoiceParameter(
        default='tau_decaymode_2_alt',
        choices=ENRICHMENT_GROUPINGS,
    )
    max_order = luigi.IntParameter(default=2)
    top_n = luigi.IntParameter(default=10)
    batch_size = luigi.IntParameter(default=1024, significant=False)

    def requires(self):
        return TrainSqueezedSingleDNNModels(
            squeezing=self.squeezing,
            reduced_weight_grouping=self.reduced_weight_grouping,
        )

    def output(self):
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'dnn'
            / self.reduced_weight_grouping
            / squeezing_label(self.squeezing)
            / f'order_{self.max_order}'
        )
        return {
            f'{category}_{process}_{artifact}': law.LocalFileTarget(
                output_root
                / taylor_category_scope(category)
                / f'top_{self.top_n}'
                / process
                / filename
            )
            for category, _ in TAYLOR_CATEGORIES
            for process in ('wjets', 'qcd', 'ttbar')
            for artifact, filename in TAYLOR_ARTIFACT_FILENAMES.items()
        }

    def run(self):
        trained_models_dir = Path(self.input()['metadata'].path).parent
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'dnn'
            / self.reduced_weight_grouping
            / squeezing_label(self.squeezing)
            / f'order_{self.max_order}'
        )
        for process in ('wjets', 'qcd', 'ttbar'):
            model_dir = trained_models_dir / process
            run_taylor_coefficient_categories(
                even_model_path=model_dir / 'fold_even',
                odd_model_path=model_dir / 'fold_odd',
                data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
                masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
                training_var_path=(
                    PROJECT_ROOT / 'configs' / 'training_variables.yaml'
                ),
                process=process,
                output_dirs={
                    category: (
                        output_root
                        / taylor_category_scope(category)
                        / f'top_{self.top_n}'
                        / process
                    )
                    for category, _ in TAYLOR_CATEGORIES
                },
                analysis_label=self.reduced_weight_grouping,
                model_type='DNN',
                max_order=self.max_order,
                top_n=self.top_n,
                batch_size=self.batch_size,
                )


class PlotSingleDNNTaylorCoefficients(law.Task):
    """Regenerate single-DNN Taylor plots from existing coefficient JSONs."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    reduced_weight_grouping = luigi.ChoiceParameter(
        default='tau_decaymode_2_alt',
        choices=ENRICHMENT_GROUPINGS,
    )
    max_order = luigi.IntParameter(default=2)
    top_n = luigi.IntParameter(default=10)

    def output(self):
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'dnn'
            / self.reduced_weight_grouping
            / squeezing_label(self.squeezing)
            / f'order_{self.max_order}'
        )
        return law.LocalFileTarget(
            output_root / f'plot_only_manifest_top_{self.top_n}.json'
        )

    def run(self):
        output_root = (
            WORKFLOW_ROOT
            / 'Taylor_coefficient_analysis'
            / 'dnn'
            / self.reduced_weight_grouping
            / squeezing_label(self.squeezing)
            / f'order_{self.max_order}'
        )
        coefficient_paths = sorted(
            output_root.glob(f'*/top_{self.top_n}/*/taylor_coefficients.json')
        )
        if not coefficient_paths:
            raise FileNotFoundError(
                "No single-DNN Taylor coefficient JSON files found under "
                f"{output_root} for top_n={self.top_n}."
            )

        rewritten = []
        for coefficient_path in coefficient_paths:
            paths = rewrite_taylor_plots(
                coefficient_path.parent,
                top_n=self.top_n,
            )
            rewritten.append({
                key: str(path)
                for key, path in paths.items()
                if key not in ('json', 'metadata')
            })

        manifest_path = Path(self.output().path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(rewritten, indent=2) + "\n")


class CalculateDNNTaylorCoefficients(CalculateSingleDNNTaylorCoefficients):
    """Backward-compatible task name for single-DNN Taylor coefficients."""


class ConvertSingleDNNModels(law.Task):
    """Apply inclusive normalization and export single-DNN FF models."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    reduced_weight_grouping = luigi.ChoiceParameter(
        default='tau_decaymode_2_alt',
        choices=ENRICHMENT_GROUPINGS,
    )

    def requires(self):
        return TrainSqueezedSingleDNNModels(
            squeezing=self.squeezing,
            reduced_weight_grouping=self.reduced_weight_grouping,
        )

    def output(self):
        output_dir = (
            WORKFLOW_ROOT
            / 'CombinedModelsSingleDNN'
            / self.reduced_weight_grouping
            / squeezing_label(self.squeezing)
        )
        outputs = {
            f'{process}_torch': law.LocalFileTarget(
                output_dir
                / FF_MODEL_OUTPUT_NAMES[process]
                / 'torch_model'
                / 'model_weights.pth'
            )
            for process in ('wjets', 'qcd', 'ttbar')
        }
        outputs.update({
            f'{process}_onnx': law.LocalFileTarget(
                output_dir
                / FF_MODEL_OUTPUT_NAMES[process]
                / 'onnx_model'
                / 'model.onnx'
            )
            for process in ('wjets', 'qcd', 'ttbar')
        })
        outputs['normalization_constants'] = law.LocalFileTarget(
            output_dir / 'normalization_constants.json'
        )
        return outputs

    def run(self):
        trained_models_dir = Path(self.input()['metadata'].path).parent
        convert_single_dnn_models(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            trained_models_dir=trained_models_dir,
            reduced_weight_dir=WORKFLOW_FEATURE_ROOT / 'reduced_dataset',
            reduced_weight_grouping=self.reduced_weight_grouping,
            output_dir=(
                WORKFLOW_ROOT
                / 'CombinedModelsSingleDNN'
                / self.reduced_weight_grouping
                / squeezing_label(self.squeezing)
            ),
        )


class CalculateSingleDNNFakeFactors(law.Task):
    """Calculate and store inclusive single-DNN fake factors."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    reduced_weight_grouping = luigi.ChoiceParameter(
        default='tau_decaymode_2_alt',
        choices=ENRICHMENT_GROUPINGS,
    )
    batch_size = luigi.IntParameter(default=65536, significant=False)
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )

    def requires(self):
        return ConvertSingleDNNModels(
            squeezing=self.squeezing,
            reduced_weight_grouping=self.reduced_weight_grouping,
        )

    def output(self):
        output_dir = (
            WORKFLOW_FEATURE_ROOT
            / 'fake_factors_single_dnn'
            / self.reduced_weight_grouping
            / squeezing_label(self.squeezing)
        )
        return {
            'features': law.LocalFileTarget(
                output_dir / 'fake_factors_single_dnn.feather'
            ),
            'schema': law.LocalFileTarget(
                output_dir / '.schema_v2_unique_feature_names'
            ),
        }

    def run(self):
        calculate_single_dnn_fake_factors(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_variables_path=(
                PROJECT_ROOT / 'configs' / 'training_variables.yaml'
            ),
            combined_models_dir=Path(
                self.input()['normalization_constants'].path
            ).parent,
            feature_store_path=self.output()['features'].path,
            feature_registry_path=(
                WORKFLOW_FEATURE_ROOT / 'feature_registry.json'
            ),
            process_fractions_path=self.process_fractions_path,
            batch_size=self.batch_size,
            feature_suffix=single_dnn_feature_suffix(
                self.squeezing,
                self.reduced_weight_grouping,
            ),
        )
        Path(self.output()['schema'].path).write_text(
            "three process FFs and one combined FF from inclusive DNNs\n"
        )


class PlotSingleDNNFakeFactorDistributions(law.Task):
    """Create inclusive closure and FF-distribution plots for single DNNs."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    reduced_weight_grouping = luigi.ChoiceParameter(
        default='tau_decaymode_2_alt',
        choices=ENRICHMENT_GROUPINGS,
    )
    variable_set = luigi.Parameter(default='variables_set_small')
    process_fractions_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/fake_factors_et.json.gz"
        )
    )
    classic_corrections_path = law.Parameter(
        default=(
            "/work/mmoser/TauFakeFactors.back/workdir/"
            "ff_2026_01_19_check_variable/2018/FF_corrections_et.json.gz"
        )
    )

    def requires(self):
        return {
            'dnn': CalculateSingleDNNFakeFactors(
                squeezing=self.squeezing,
                reduced_weight_grouping=self.reduced_weight_grouping,
                process_fractions_path=self.process_fractions_path,
            ),
            'classic': CalculateClassicFakeFactors(
                fake_factors_path=self.process_fractions_path,
                corrections_path=self.classic_corrections_path,
            ),
            'reduced': ReducedDataset(),
        }

    def output(self):
        return law.LocalFileTarget(
            WORKFLOW_ROOT
            / 'plots'
            / 'fake_factors_single_dnn'
            / self.reduced_weight_grouping
            / squeezing_label(self.squeezing)
            / self.variable_set
            / 'manifest_subsets_v3.json'
        )

    def run(self):
        create_single_dnn_fake_factor_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            feature_paths=(
                self.input()['dnn']['features'].path,
                self.input()['classic']['features'].path,
            ),
            reduced_weight_paths=(
                Path(self.input()['reduced'][process].path)
                / f'reduced_weight_{self.reduced_weight_grouping}.feather'
                for process in ('wjets', 'qcd')
            ),
            plotting_config_path=PROJECT_ROOT / 'configs' / 'plotting.yaml',
            labels_path=PROJECT_ROOT / 'configs' / 'labels.yaml',
            output_dir=Path(self.output().path).parent,
            manifest_path=self.output().path,
            reduced_weight_grouping=self.reduced_weight_grouping,
            variable_set=self.variable_set,
            feature_suffix=single_dnn_feature_suffix(
                self.squeezing,
                self.reduced_weight_grouping,
            ),
        )


class PlotSingleDNNHighFakeFactorDistributions(law.Task):
    """Plot high-range single-DNN FF distributions only."""

    squeezing = luigi.OptionalFloatParameter(default=None)
    reduced_weight_grouping = luigi.ChoiceParameter(
        default='tau_decaymode_2_alt',
        choices=ENRICHMENT_GROUPINGS,
    )
    value_min = luigi.FloatParameter(default=1.0)
    value_max = luigi.FloatParameter(default=10.0)
    n_bins = luigi.IntParameter(default=90)

    def requires(self):
        return CalculateSingleDNNFakeFactors(
            squeezing=self.squeezing,
            reduced_weight_grouping=self.reduced_weight_grouping,
        )

    def output(self):
        return law.LocalFileTarget(
            WORKFLOW_ROOT
            / 'plots'
            / 'fake_factors_single_dnn'
            / self.reduced_weight_grouping
            / squeezing_label(self.squeezing)
            / f'high_range_{self.value_min:g}_{self.value_max:g}'
            / 'manifest.json'
        )

    def run(self):
        create_single_dnn_distribution_plots(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            feature_path=self.input()['features'].path,
            output_dir=Path(self.output().path).parent,
            manifest_path=self.output().path,
            value_min=self.value_min,
            value_max=self.value_max,
            n_bins=self.n_bins,
            feature_suffix=single_dnn_feature_suffix(
                self.squeezing,
                self.reduced_weight_grouping,
            ),
        )


class TrainConditionalNF(law.Task):
    """Train two-fold conditional normalizing flows on reduced datasets."""

    test_size = luigi.FloatParameter(default=0.25)
    random_state = luigi.IntParameter(default=42)

    def requires(self):
        return ReducedDataset()

    def output(self):
        output_root = WORKFLOW_ROOT / 'Training_results_NF'
        return {
            f"{grouping}_{process}_{region}_{fold}": law.LocalFileTarget(
                output_root
                / grouping
                / process
                / region
                / fold
                / 'model_checkpoint.pth'
            )
            for grouping in ENRICHMENT_GROUPINGS
            for process in ('wjets', 'qcd')
            for region in ('AR-like', 'SR-like')
            for fold in ('fold_even', 'fold_odd')
        }

    def run(self):
        train_conditional_flows(
            data_path=WORKFLOW_DATA_ROOT / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            variables_path=PROJECT_ROOT / 'configs' / 'training_variables_nf.yaml',
            config_path=PROJECT_ROOT / 'configs' / 'config_NF.yaml',
            output_root=WORKFLOW_ROOT / 'Training_results_NF',
            test_size=self.test_size,
            random_state=self.random_state,
        )
