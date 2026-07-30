import math
from typing import Optional


def squeezing_feature_suffix(squeezing: Optional[float]) -> str:
    if squeezing is None:
        return ""
    if not 0.0 < squeezing < 1.0:
        raise ValueError(
            f"squeezing must be between 0 and 1 (exclusive), got {squeezing}"
        )

    rounded = round(float(squeezing), 2)
    if not math.isclose(squeezing, rounded, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Feature names support squeezing values with at most two "
            f"decimal places, got {squeezing}."
        )
    return f"_{int(round(rounded * 100)):02d}"


def single_dnn_feature_suffix(squeezing, reduced_weight_grouping):
    grouping_part = (
        ""
        if reduced_weight_grouping == "tau_decaymode_2_alt"
        else f"_{reduced_weight_grouping}"
    )
    return f"{grouping_part}{squeezing_feature_suffix(squeezing)}"


def ff_name(squeezing: Optional[float] = 0.99) -> str:
    """Name of the default DRSR- and non-closure-corrected FF."""
    return f"ff{squeezing_feature_suffix(squeezing)}"


def uncorrected_ff_name(squeezing: Optional[float] = 0.99) -> str:
    return f"ff_uncorrected{squeezing_feature_suffix(squeezing)}"


def nonclosure_ff_name(squeezing: Optional[float] = 0.99) -> str:
    return f"ff_nonclosure{squeezing_feature_suffix(squeezing)}"


def drsr_ff_name(squeezing: Optional[float] = 0.99) -> str:
    return f"ff_drsr{squeezing_feature_suffix(squeezing)}"


def classic_fraction_ff_name(squeezing: Optional[float] = None) -> str:
    return f"ff_cf{squeezing_feature_suffix(squeezing)}"


def drsr_correction_name(
    process: str,
    squeezing: Optional[float] = 0.99,
) -> str:
    if process not in ("wjets", "qcd", "ttbar"):
        raise ValueError(f"Unsupported DRSR correction process: {process}")
    return f"correction_drsr_{process}{squeezing_feature_suffix(squeezing)}"
