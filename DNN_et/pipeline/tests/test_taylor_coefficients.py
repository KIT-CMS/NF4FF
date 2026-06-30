from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch as t


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from classes import DNN, FoldCombinedDNN, GroupedDNN
from taylor_coefficient_analysis import (
    _analysis_arrays,
    calculate_taylor_coefficients,
    calculate_taylor_coefficients_manually,
    write_taylor_artifacts,
)


def _model(shift, scale):
    model = DNN(
        input_nodes=2,
        hidden_nodes=[4],
        output_nodes=1,
        dropout=[0.0],
        activation="Tanh",
        output_activation="Sigmoid",
        input_names=["x", "y"],
    )
    model.initialize_scaler(t.tensor(shift), t.tensor(scale))
    return model.eval()


def test_temporary_scaler_extraction_matches_manual_fold_calculation():
    t.manual_seed(4)
    model = FoldCombinedDNN(
        even_model=_model([1.0, -2.0], [2.0, 0.5]),
        odd_model=_model([-0.5, 3.0], [1.5, 4.0]),
        fold_id_name="event",
    ).eval()
    original = deepcopy(model.state_dict())
    features = t.tensor([
        [0.0, -1.0],
        [2.0, 1.0],
        [-3.0, 4.0],
        [1.5, 2.5],
    ])
    events = t.tensor([2, 3, 4, 5])

    extracted = calculate_taylor_coefficients(
        model,
        features,
        ["x", "y"],
        event_ids=events,
        batch_size=2,
    )
    manual = calculate_taylor_coefficients_manually(
        model,
        features,
        ["x", "y"],
        event_ids=events,
        batch_size=2,
    )

    for order in ("first_order", "second_order"):
        assert extracted[order].keys() == manual[order].keys()
        for key in extracted[order]:
            assert np.isclose(
                extracted[order][key],
                manual[order][key],
                rtol=1e-6,
                atol=1e-8,
            )

    for name, value in model.state_dict().items():
        assert t.equal(value, original[name])


def test_manual_grouped_scaler_accepts_integer_bounds_with_float_inputs():
    t.manual_seed(5)
    default_model = _model([1.0, -2.0], [2.0, 0.5])
    model = GroupedDNN(
        grouping={0: ((0,), (1,))},
        default_model=default_model,
    ).eval()
    features = t.tensor([
        [0.0, -1.0],
        [1.0, 1.0],
        [0.0, 4.0],
        [1.0, 2.5],
    ], dtype=t.float32)

    extracted = calculate_taylor_coefficients(
        model,
        features,
        ["x", "y"],
        batch_size=2,
    )
    manual = calculate_taylor_coefficients_manually(
        model,
        features,
        ["x", "y"],
        batch_size=2,
    )

    for order in ("first_order", "second_order"):
        for key in extracted[order]:
            assert np.isclose(
                extracted[order][key],
                manual[order][key],
                rtol=1e-6,
                atol=1e-8,
            )


def test_written_json_uses_legacy_flat_format(tmp_path):
    coefficients = {
        "first_order": {"x": 1.0, "y": 2.0},
        "second_order": {
            "x,x": 0.5,
            "x,y": 0.25,
            "y,y": 0.125,
        },
    }

    paths = write_taylor_artifacts(coefficients, tmp_path)

    assert json.loads(paths["json"].read_text()) == {
        "x": 1.0,
        "y": 2.0,
        "x,x": 0.5,
        "x,y": 0.25,
        "y,y": 0.125,
    }
    assert paths["style"].exists()


@pytest.mark.parametrize(
    ("category", "expected_events"),
    (
        ("inclusive", [10, 11, 12, 13]),
        ("njets_eq_0", [10]),
        ("njets_eq_1", [11]),
        ("njets_ge_2", [12, 13]),
        ("tau_decaymode_2_eq_0", [10]),
        ("tau_decaymode_2_eq_11", [13]),
        ("tau_decaymode_2_in_0_1", [10, 11]),
        ("tau_decaymode_2_in_10_11", [12, 13]),
    ),
)
def test_analysis_arrays_select_taylor_category(category, expected_events):
    frame = np.asarray([
        [10, 0.1, 0, 0],
        [11, 0.2, 1, 1],
        [12, 0.3, 2, 10],
        [13, 0.4, 4, 11],
    ], dtype=np.float32)

    _, events = _analysis_arrays(
        frame,
        ["x", "njets", "tau_decaymode_2"],
        category,
    )

    assert events.tolist() == expected_events
