from pathlib import Path
import sys

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from uncertainty_model_workflow import (
    plot_taylor_ensemble,
    plot_taylor_method_comparison,
    summarize_taylor_ensemble,
    uncertainty_taylor_artifact_paths,
    uncertainty_taylor_coefficient_paths,
    uncertainty_taylor_plot_paths,
)


def test_taylor_ensemble_summary_ranks_by_mean_and_uses_population_std():
    coefficients = {
        "100": {
            "first_order": {"x": 1.0, "y": 4.0},
            "second_order": {"x,x": 3.0},
        },
        "101": {
            "first_order": {"x": 3.0, "y": 2.0},
            "second_order": {"x,x": 5.0},
        },
    }

    summary = summarize_taylor_ensemble(coefficients, top_n=2)

    assert [entry["name"] for entry in summary] == ["x,x", "y"]
    assert summary[0]["order"] == "second_order"
    assert np.isclose(summary[0]["mean"], 4.0)
    assert np.isclose(summary[0]["std"], 1.0)


def test_taylor_ensemble_plot_writes_png_and_pdf(tmp_path):
    summary = [
        {"name": "x", "order": "first_order", "mean": 2.0, "std": 0.5},
        {"name": "x,x", "order": "second_order", "mean": 1.0, "std": 0.2},
    ]

    for extension in ("png", "pdf"):
        output_path = tmp_path / f"plot.{extension}"
        plot_taylor_ensemble(summary, output_path, title="Test")
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_taylor_method_comparison_plot_writes_png(tmp_path):
    trained = [
        {"name": "x", "order": "first_order", "mean": 2.0, "std": 0.5},
        {"name": "x,x", "order": "second_order", "mean": 1.0, "std": 0.2},
    ]
    dropout = [
        {"name": "x,x", "order": "second_order", "mean": 1.2, "std": 0.3},
        {"name": "x", "order": "first_order", "mean": 1.8, "std": 0.4},
    ]
    output_path = tmp_path / "comparison.png"

    plot_taylor_method_comparison(
        trained,
        dropout,
        output_path,
        top_n=2,
        title="Comparison",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_uncertainty_taylor_artifact_paths_are_process_scoped(tmp_path):
    paths = uncertainty_taylor_artifact_paths(tmp_path, "wjets", top_n=10)

    assert paths["models_coefficients"] == (
        tmp_path / "wjets" / "trained_models_coefficients.json"
    )
    assert paths["models_png"] == (
        tmp_path / "wjets" / "trained_models_top_10.png"
    )
    assert paths["comparison_pdf"] == (
        tmp_path / "wjets" / "method_comparison_top_10.pdf"
    )


def test_coefficient_and_plot_artifacts_are_separate(tmp_path):
    coefficient_paths = uncertainty_taylor_coefficient_paths(tmp_path, "qcd")
    plot_paths = uncertainty_taylor_plot_paths(tmp_path, "qcd", top_n=5)

    assert set(coefficient_paths) == {
        "models_coefficients",
        "dropout_coefficients",
    }
    assert "models_summary" not in coefficient_paths
    assert "models_coefficients" not in plot_paths
    assert plot_paths["models_summary"].name == "trained_models_top_5.json"
