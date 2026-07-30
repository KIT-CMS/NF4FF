import argparse
import json
"""QCD yield-correction calculation."""

from pathlib import Path

import numpy as np

from data.handling import load_data
from core.paths import CONFIG_ROOT


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "dataframe_complete.feather"
DEFAULT_MASKS_PATH = CONFIG_ROOT / "selections.yaml"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "Law_workflow_results" / "qcd_yield_corrections.json"
)

PROCESS_COMPONENTS = (
    ("wjets", "W+jets", "#e76300"),
    ("embedding", r"$\tau$ embedded", "#ffa90e"),
    ("diboson", "Diboson", "#94a4a2"),
    ("DYjets", r"DY+jets", "#3f90da"),
    ("ST", "Single top", "#717581"),
    ("ttbar", r"$t\bar{t}$", "#832db6"),
)

NJET_BINS = (
    ("njets_0", "njets == 0", lambda frame: frame["njets"] == 0),
    ("njets_1", "njets == 1", lambda frame: frame["njets"] == 1),
    ("njets_ge_2", "njets >= 2", lambda frame: frame["njets"] >= 2),
)


def _weighted_yield(frame, njet_selector):
    selected = frame.loc[njet_selector(frame)]
    weights = selected["weight"].to_numpy(dtype=np.float64)
    return float(weights.sum(dtype=np.float64))


def _qcd_subtracted_yield(df, region_name, njet_selector):
    data_frame = getattr(df.data, region_name).events
    data_yield = _weighted_yield(data_frame, njet_selector)

    components = {}
    mc_yield = 0.0
    for process, _, _ in PROCESS_COMPONENTS:
        process_frame = getattr(getattr(df, process), region_name).events
        process_yield = _weighted_yield(process_frame, njet_selector)
        components[process] = process_yield
        mc_yield += process_yield

    return {
        "data": data_yield,
        "mc_components": components,
        "mc_total": mc_yield,
        "qcd": data_yield - mc_yield,
    }


def calculate_qcd_yield_corrections(df):
    corrections = {}

    for bin_name, label, selector in NJET_BINS:
        os_yield = _qcd_subtracted_yield(df, "AR", selector)
        ss_yield = _qcd_subtracted_yield(df, "AR_SS", selector)

        if ss_yield["qcd"] == 0:
            raise ZeroDivisionError(
                f"{label}: cannot calculate OS/SS correction because "
                "the subtracted SS QCD yield is zero."
            )

        corrections[bin_name] = {
            "label": label,
            "yield_OS": os_yield,
            "yield_SS": ss_yield,
            "correction": os_yield["qcd"] / ss_yield["qcd"],
        }

    return corrections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--masks-path", type=Path, default=DEFAULT_MASKS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    df = load_data(args.data_path, args.masks_path)
    corrections = calculate_qcd_yield_corrections(df)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(corrections, handle, indent=2)

    for bin_name, result in corrections.items():
        print(
            f"{bin_name}: correction={result['correction']:.8g} "
            f"(OS={result['yield_OS']['qcd']:.8g}, "
            f"SS={result['yield_SS']['qcd']:.8g})"
        )
    print(f"Saved QCD yield corrections to {args.output_path}")


if __name__ == "__main__":
    main()
