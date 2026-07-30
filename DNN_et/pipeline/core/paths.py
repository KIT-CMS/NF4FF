from pathlib import Path
from typing import Optional

from .methods import FFMethod


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "pipeline" / "config"
WORKFLOW_ROOT = PROJECT_ROOT / "Law_workflow_results"
DATA_ROOT = WORKFLOW_ROOT / "data"
FEATURE_ROOT = DATA_ROOT / "features"
MODEL_ROOT = WORKFLOW_ROOT / "models"
PLOT_ROOT = WORKFLOW_ROOT / "plots"
CORRECTION_ROOT = PROJECT_ROOT / "corrections" / "fake_factors"


def squeezing_label(squeezing: Optional[float]) -> str:
    return "unsqueezed" if squeezing is None else f"{squeezing:.4f}"


def drsr_loss_limit_label(loss_limit: float) -> str:
    value = str(float(loss_limit)).replace(".", "p").replace("-", "m")
    return f"loss_squeeze_pm{value}"


def drsr_limit_subdir(path, loss_limit):
    return Path(path) / drsr_loss_limit_label(loss_limit)


def ff_feature_dir(method, squeezing, loss_limit=None):
    method = FFMethod.parse(method)
    path = FEATURE_ROOT / "fake_factors" / method.value / squeezing_label(squeezing)
    return drsr_limit_subdir(path, loss_limit) if method.uses_drsr else path


def ff_plot_dir(kind, method, squeezing, loss_limit=None):
    method = FFMethod.parse(method)
    path = (
        PLOT_ROOT
        / "fake_factors"
        / kind
        / method.value
        / squeezing_label(squeezing)
    )
    return drsr_limit_subdir(path, loss_limit) if method.uses_drsr else path
