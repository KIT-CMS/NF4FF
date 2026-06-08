import law
from pathlib import Path

from BuildDataset import build_dataset
from enrichment import train_enrichment_wjets, train_enrichment_qcd
from ReducedDataset import reduced_data_wjets, reduced_data_qcd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
        return law.LocalDirectoryTarget(base)

    def run(self):
        if self.trainer is None:
            raise RuntimeError("No trainer function configured for this process task")

        print("TRAIN INPUT:", self.input().path)
        result = self.trainer(self.input().path)
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
        }

    def run(self):
        reduced_data_wjets()
        reduced_data_qcd()
        print("REDUCED DATASET WJETS OUTPUT:", self.output()['wjets'].path)
        print("REDUCED DATASET QCD OUTPUT:", self.output()['qcd'].path)


