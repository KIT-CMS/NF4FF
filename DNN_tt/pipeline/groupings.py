import math


GROUPING_NAMES = (
    "tau_decaymode_2",
    "tau_decaymode_2_alt",
    "njets",
)

GROUPING_SOURCES = {
    "tau_decaymode": "tau_decaymode_2",
    "tau_decaymode_2": "tau_decaymode_2",
    "tau_decaymode_2_alt": "tau_decaymode_2",
    "njets": "njets",
}

GROUPING_BOUNDS = {
    "tau_decaymode": ((0,), (1,), (10,), (11,)),
    "tau_decaymode_2": ((0,), (1,), (10,), (11,)),
    "tau_decaymode_2_alt": ((0, 2), (10, 12)),
    "njets": ((0,), (1,), (2, 1000)),
}

TTBAR_NJETS_BOUNDS = ((0, 1), (2, 1000))

GROUPING_SUFFIXES = {
    "tau_decaymode": "",
    "tau_decaymode_2": "",
    "tau_decaymode_2_alt": "_tau_decaymode_2_alt",
    "njets": "_njets",
}


def grouping_source(grouping):
    try:
        return GROUPING_SOURCES[grouping]
    except KeyError as error:
        raise ValueError(f"Unsupported grouping: {grouping}") from error


def grouping_bounds(grouping, process=None):
    if grouping == "njets" and process == "ttbar":
        return TTBAR_NJETS_BOUNDS
    try:
        return GROUPING_BOUNDS[grouping]
    except KeyError as error:
        raise ValueError(f"Unsupported grouping: {grouping}") from error


def grouping_suffix(grouping):
    try:
        return GROUPING_SUFFIXES[grouping]
    except KeyError as error:
        raise ValueError(f"Unsupported grouping: {grouping}") from error


def squeezing_feature_suffix(squeezing):
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
