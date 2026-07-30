from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "selections.yaml"


def _config():
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_dr_qcd_fractions_region_is_preselected_ar_opposite_sign():
    config = _config()

    assert config["regions"]["DR_qcd_fractions"] == [
        "preselection",
        "DR_qcd_fractions",
    ]
    assert config["regions"]["AR_SS"] == ["preselection", "AR_SS"]
    assert set(config["masks"]["DR_qcd_fractions"]) == set(config["masks"]["AR_SS"])


def test_dr_qcd_fractions_no_signs_region_uses_preselection():
    config = _config()

    assert config["regions"]["DR_qcd_fractions_no_signs"] == [
        "preselection",
        "DR_qcd_fractions_no_signs",
    ]
