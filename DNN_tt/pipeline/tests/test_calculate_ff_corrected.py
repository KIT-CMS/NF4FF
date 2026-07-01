from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

import calculate_ff_corrected as corrected


class _Input:
    def __init__(self, name):
        self.name = name


class _CompoundCorrection:
    def __init__(self, factor):
        self.factor = factor
        self.inputs = [
            _Input("tau_decaymode_2"),
            _Input("njets"),
            _Input("syst"),
        ]

    def evaluate(self, tau_decaymode, njets, syst):
        assert syst == "nominal"
        return np.full(len(tau_decaymode), self.factor)


class _Corrections:
    def __init__(self):
        self.compound = {
            "Wjets_compound_correction": _CompoundCorrection(2.0),
            "QCD_compound_correction": _CompoundCorrection(3.0),
            "ttbar_compound_correction": _CompoundCorrection(4.0),
        }


class _ProcessFractions:
    _fractions = {"Wjets": 0.5, "QCD": 0.3, "ttbar": 0.2}

    def evaluate(self, process, mt_1, njets, syst):
        assert syst == "nominal"
        return np.full(len(mt_1), self._fractions[process])


def test_calculate_corrected_fake_factors_adds_column(monkeypatch):
    seen = {}

    def fake_load(correction_dir):
        seen["correction_dir"] = correction_dir
        return {"process_fractions": _ProcessFractions()}, _Corrections()

    monkeypatch.setattr(corrected, "_load_correction_sets", fake_load)
    frame = pd.DataFrame({
        "mt_1": [20.0, 30.0],
        "njets": [0, 2],
        "tau_decaymode_2": [0, 10],
        "ff_dnn_wjets": [1.0, 2.0],
        "ff_dnn_qcd": [2.0, 3.0],
        "ff_dnn_ttbar": [3.0, 4.0],
    })

    output_name = corrected.calculate_corrected_fake_factors(
        frame,
        "/corrections",
    )

    assert output_name == "ff_dnn_corrected"
    np.testing.assert_allclose(
        frame[output_name],
        [
            0.5 * 2.0 + 0.3 * 6.0 + 0.2 * 12.0,
            0.5 * 4.0 + 0.3 * 9.0 + 0.2 * 16.0,
        ],
    )
    assert seen["correction_dir"] == Path(
        "/corrections/unsqueezed/tau_decaymode_2"
    )


@pytest.mark.parametrize(
    ("grouping", "squeezing", "source_suffix", "output_name", "directory"),
    (
        (
            "tau_decaymode_2_alt",
            None,
            "_tau_decaymode_2_alt",
            "ff_dnn_corrected_tau_decaymode_2_alt",
            "unsqueezed/tau_decaymode_2_alt",
        ),
        (
            "njets",
            0.98,
            "_njets_98",
            "ff_dnn_corrected_njets_98",
            "0.9800/njets",
        ),
    ),
)
def test_calculate_corrected_fake_factors_grouping_and_squeezing_names(
    monkeypatch,
    grouping,
    squeezing,
    source_suffix,
    output_name,
    directory,
):
    seen = {}

    def fake_load(correction_dir):
        seen["correction_dir"] = correction_dir
        return {"process_fractions": _ProcessFractions()}, _Corrections()

    monkeypatch.setattr(corrected, "_load_correction_sets", fake_load)
    frame = pd.DataFrame({
        "mt_1": [20.0],
        "njets": [2],
        "tau_decaymode_2": [10],
        f"ff_dnn_wjets{source_suffix}": [1.0],
        f"ff_dnn_qcd{source_suffix}": [2.0],
        f"ff_dnn_ttbar{source_suffix}": [3.0],
    })

    result = corrected.calculate_corrected_fake_factors(
        frame,
        "/corrections",
        grouping=grouping,
        squeezing=squeezing,
    )

    assert result == output_name
    assert output_name in frame
    assert seen["correction_dir"] == Path("/corrections") / directory


def test_calculate_corrected_fake_factors_rejects_unknown_grouping():
    with pytest.raises(ValueError, match="Unsupported corrected FF grouping"):
        corrected.calculate_corrected_fake_factors(
            pd.DataFrame(),
            "/corrections",
            grouping="unknown",
        )
