from pathlib import Path
import sys
import types

import pandas as pd
import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))


classes_stub = types.ModuleType("classes")
classes_stub.DNN = object
classes_stub.FoldCombinedDNN = object
classes_stub.load_data = None
classes_stub.load_variables = None
classes_stub.save_model = None
sys.modules.setdefault("classes", classes_stub)

lightning_stub = types.ModuleType("lightning")
lightning_pytorch_stub = types.ModuleType("lightning.pytorch")
lightning_pytorch_stub.LightningDataModule = object
lightning_pytorch_stub.LightningModule = object
callbacks_stub = types.ModuleType("lightning.pytorch.callbacks")
callbacks_stub.ModelCheckpoint = object
callbacks_stub.EarlyStopping = object
loggers_stub = types.ModuleType("lightning.pytorch.loggers")
loggers_stub.CSVLogger = object
lightning_stub.pytorch = lightning_pytorch_stub
sys.modules.setdefault("lightning", lightning_stub)
sys.modules.setdefault("lightning.pytorch", lightning_pytorch_stub)
sys.modules.setdefault("lightning.pytorch.callbacks", callbacks_stub)
sys.modules.setdefault("lightning.pytorch.loggers", loggers_stub)

from multiclass_classification import (
    QCD_FRACTION_WEIGHT_COLUMN,
    _qcd_yield_correction_array,
    _weight_array,
    validate_qcd_fraction_weights,
)


def test_validate_qcd_fraction_weights_accepts_finite_region_weights():
    frame = pd.DataFrame({QCD_FRACTION_WEIGHT_COLUMN: [0.4, 1.0, 2.5]})

    validate_qcd_fraction_weights(frame)


def test_validate_qcd_fraction_weights_rejects_partly_undefined_region_weights():
    frame = pd.DataFrame({QCD_FRACTION_WEIGHT_COLUMN: [0.4, None, 2.5]})

    with pytest.raises(ValueError, match="contains non-finite weights for 1/3"):
        validate_qcd_fraction_weights(frame)


def test_validate_qcd_fraction_weights_rejects_all_undefined_region_weights():
    frame = pd.DataFrame({QCD_FRACTION_WEIGHT_COLUMN: [None, None]})

    with pytest.raises(ValueError, match="contains non-finite weights for 2/2"):
        validate_qcd_fraction_weights(frame)


def test_weight_array_uses_requested_column_values():
    frame = pd.DataFrame({"weight": [0.5, 2.0, 3.5]})

    assert _weight_array(frame, "weight", "test").tolist() == [0.5, 2.0, 3.5]


def test_weight_array_rejects_missing_column():
    frame = pd.DataFrame({"other": [1.0]})

    with pytest.raises(ValueError, match="missing event-weight column weight"):
        _weight_array(frame, "weight", "test")


def test_qcd_yield_correction_array_maps_njets_bins():
    frame = pd.DataFrame({"njets": [0, 1, 2, 3]})
    corrections = {
        "njets_0": {"correction": 0.5},
        "njets_1": {"correction": 1.5},
        "njets_ge_2": {"correction": 2.0},
    }

    assert _qcd_yield_correction_array(frame, corrections).tolist() == [
        0.5,
        1.5,
        2.0,
        2.0,
    ]
