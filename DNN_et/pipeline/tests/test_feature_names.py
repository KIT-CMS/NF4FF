from pathlib import Path
import sys

import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from groupings import single_dnn_feature_suffix, squeezing_feature_suffix


@pytest.mark.parametrize(
    ("squeezing", "expected"),
    (
        (None, ""),
        (0.9, "_90"),
        (0.95, "_95"),
        (0.98, "_98"),
        (0.99, "_99"),
    ),
)
def test_squeezing_feature_suffix(squeezing, expected):
    assert squeezing_feature_suffix(squeezing) == expected


def test_squeezing_feature_suffix_rejects_ambiguous_precision():
    with pytest.raises(ValueError, match="at most two decimal places"):
        squeezing_feature_suffix(0.985)


@pytest.mark.parametrize(
    ("squeezing", "grouping", "expected"),
    (
        (None, "tau_decaymode_2_alt", ""),
        (0.98, "tau_decaymode_2_alt", "_98"),
        (None, "njets", "_njets"),
        (0.98, "njets", "_njets_98"),
    ),
)
def test_single_dnn_feature_suffix(squeezing, grouping, expected):
    assert single_dnn_feature_suffix(squeezing, grouping) == expected
