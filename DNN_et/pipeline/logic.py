import json
import law
import luigi
from pathlib import Path

from BuildDataset import build_dataset
from NF_training_njets import train_conditional_flows
from enrichment import (
    train_enrichment_qcd,
    train_enrichment_qcd_fractions,
    train_enrichment_wjets,
)
from ReducedDataset import reduced_data_wjets, reduced_data_qcd
from plot_reduced_training_qcd import (
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
from ff_models_to_onnx import ff_models_to_onnx
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
from calculate_ff_corrected import (
    calculate_and_store_corrected_fake_factors,
)
from plotting import (
    create_corrected_fake_factor_closure_plots,
    create_fake_factor_opposite_grouping_distribution_plots,
    create_fake_factor_plots,
    create_high_ff_closure_plots,
    create_high_fake_factor_distribution_plots,
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
        schema_path = Path(self.output()["features_schema"].path)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text("row-index keyed enrichment features\n")
        print("TRAIN OUTPUT:", result["combined_model_path"])


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
            / ".row_index_schema_v2_qcd_weights_ss"
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
