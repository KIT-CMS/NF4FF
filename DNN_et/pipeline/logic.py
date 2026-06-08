import law
import luigi
from pathlib import Path

from BuildDataset import build_dataset
from enrichment import train_enrichment_wjets, train_enrichment_qcd
from ReducedDataset import reduced_data_wjets, reduced_data_qcd
from plot_reduced_training_qcd import create_qcd_training_plots
from plot_reduced_training_wjets import create_wjets_training_plots
from training_squeezed_loss import (
    squeezing_label,
    train_squeezed_models,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENRICHMENT_GROUPINGS = ('tau_decaymode_2', 'njets')


class BuildDataset(law.Task):

    config_path = law.Parameter(default="../configs/root_data_path.yaml")

    def output(self):
        out_dir = PROJECT_ROOT / "data"
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
        base = PROJECT_ROOT / "Enrichment_models" / self.process_name
        outputs = {
            f"{grouping}_{fold}": law.LocalDirectoryTarget(
                base / grouping / fold
            )
            for grouping in ENRICHMENT_GROUPINGS
            for fold in ('fold_even', 'fold_odd')
        }
        outputs["features_schema"] = law.LocalFileTarget(
            PROJECT_ROOT
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
        result = self.trainer(self.input().path)
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


class ReducedDataset(law.Task):
    """Compute reduced datasets for both W+jets and QCD processes."""

    def requires(self):
        return {
            'wjets': TrainEnrichmentWjetsV2(),
            'qcd': TrainEnrichmentQCDV2(),
        }

    def output(self):
        return {
            'wjets': law.LocalDirectoryTarget(PROJECT_ROOT / 'data' / 'features' / 'reduced_dataset' / 'wjets'),
            'qcd': law.LocalDirectoryTarget(PROJECT_ROOT / 'data' / 'features' / 'reduced_dataset' / 'qcd'),
            'schema': law.LocalFileTarget(
                PROJECT_ROOT / 'data' / 'features' / 'reduced_dataset' / '.schema_v4'
            ),
        }

    def run(self):
        reduced_data_wjets()
        reduced_data_qcd()
        schema_path = Path(self.output()['schema'].path)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(
            "process-qualified reduced weights and signed-region NN outputs\n"
        )
        print("REDUCED DATASET WJETS OUTPUT:", self.output()['wjets'].path)
        print("REDUCED DATASET QCD OUTPUT:", self.output()['qcd'].path)


class PlotTrainingResultsWjets(law.Task):
    """Plot W+jets enrichment diagnostics for both grouping choices."""

    n_bins = luigi.IntParameter(default=20)

    def requires(self):
        return ReducedDataset()

    def output(self):
        output_dir = PROJECT_ROOT / 'plots' / 'enrichment_wjets'
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
            data_path=PROJECT_ROOT / 'data' / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            output_dir=PROJECT_ROOT / 'plots' / 'enrichment_wjets',
            n_bins=self.n_bins,
        )


class PlotTrainingResultsQCD(law.Task):
    """Plot QCD enrichment diagnostics for both grouping choices."""

    n_bins = luigi.IntParameter(default=20)

    def requires(self):
        return ReducedDataset()

    def output(self):
        output_dir = PROJECT_ROOT / 'plots' / 'enrichment_qcd'
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
            data_path=PROJECT_ROOT / 'data' / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            output_dir=PROJECT_ROOT / 'plots' / 'enrichment_qcd',
            n_bins=self.n_bins,
        )


class TrainSqueezedModels(law.Task):
    """Train grouped FF models with an optional output squeezing probability."""

    squeezing = luigi.OptionalFloatParameter(default=None)

    def requires(self):
        return ReducedDataset()

    def output(self):
        return law.LocalFileTarget(
            PROJECT_ROOT
            / 'Training_results_squeezed'
            / squeezing_label(self.squeezing)
            / '.complete'
        )

    def run(self):
        train_squeezed_models(
            squeezing=self.squeezing,
            data_path=PROJECT_ROOT / 'data' / 'dataframe_complete.feather',
            masks_path=PROJECT_ROOT / 'configs' / 'masks.yaml',
            training_var_path=PROJECT_ROOT / 'configs' / 'training_variables.yaml',
            nn_config_path=PROJECT_ROOT / 'configs' / 'DNN.yaml',
            checkpoint_dir=PROJECT_ROOT / 'Training_results_squeezed',
        )

        output_path = Path(self.output().path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            f"squeezing_probability={self.squeezing}\n"
        )
