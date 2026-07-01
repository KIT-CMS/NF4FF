from pathlib import Path
import sys

import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from plotting import (
    CLOSURE_ONLY_SUBSETS,
    GROUPING_NAMES,
    HIGH_FF_CLOSURE_GROUPINGS,
    _opposite_distribution_grouping,
    _plot_subset,
)


class Frame:
    def __init__(self, events):
        self.events = events

    def subset(self, mask):
        return Frame(self.events.loc[mask])


def test_closure_only_decay_mode_subsets_select_combined_modes():
    frame = Frame(pd.DataFrame({"tau_decaymode_2": [0, 1, 10, 11, 2]}))

    selected_modes = {
        name: _plot_subset(frame, selection).events["tau_decaymode_2"].tolist()
        for name, _, selection in CLOSURE_ONLY_SUBSETS
        if name.startswith("tau_decaymode_2_in_")
    }

    assert selected_modes == {
        "tau_decaymode_2_in_0_1": [0, 1],
        "tau_decaymode_2_in_10_11": [10, 11],
    }


def test_high_ff_closure_groupings_match_category():
    assert HIGH_FF_CLOSURE_GROUPINGS["inclusive"] == GROUPING_NAMES
    assert HIGH_FF_CLOSURE_GROUPINGS["njets_eq_0"] == ("njets",)
    assert HIGH_FF_CLOSURE_GROUPINGS["tau_decaymode_2_eq_0"] == (
        "tau_decaymode_2",
    )
    assert HIGH_FF_CLOSURE_GROUPINGS["tau_decaymode_2_in_0_1"] == (
        "tau_decaymode_2_alt",
    )


def test_ff_distribution_uses_opposite_grouping_for_split():
    assert _opposite_distribution_grouping("njets") == "tau_decaymode_2"
    assert _opposite_distribution_grouping("tau_decaymode_2") == "njets"
    assert _opposite_distribution_grouping("tau_decaymode_2_alt") == "njets"
