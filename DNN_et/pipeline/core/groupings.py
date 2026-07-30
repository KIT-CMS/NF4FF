from core.names import single_dnn_feature_suffix, squeezing_feature_suffix


GROUPING_NAMES = (
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
