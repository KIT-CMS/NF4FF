from pathlib import Path
import pandas as pd
import uproot
import yaml

from classes import get_class_weights


PROCESSES = {
    "Wjets": 1,
    "data": 0,
    "diboson_J": 2,
    "diboson_L": 3,
    "DYjets_J": 4,
    "DYjets_L": 5,
    "ST_J": 6,
    "ST_L": 7,
    "ttbar_J": 8,
    "ttbar_L": 9,

    "diboson_T": 11,
    "DYjets_T": 12,
    "ST_T": 13,
    "ttbar_T": 14,
}
REQUIRED_TRUE_TAU_PROCESSES = {
    "diboson_T": 11,
    "DYjets_T": 12,
    "ST_T": 13,
    "ttbar_T": 14,
}
EVENT_COLUMN_CANDIDATES = (
    "event",
    "evt",
    "Event",
    "eventNumber",
    "event_number",
)


def _ensure_event_column(df: pd.DataFrame, process_id: int, process_name: str):
    if "event" in df.columns:
        return df

    for candidate in EVENT_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return df.rename(columns={candidate: "event"})

    available = ", ".join(map(str, df.columns[:80]))
    raise KeyError(
        f"{process_name}.root does not contain an event branch with one of "
        f"the expected names {EVENT_COLUMN_CANDIDATES}. Available columns "
        f"start with: {available}"
    )


def build_dataset(config_path: str) -> pd.DataFrame:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    directories = cfg.get("directories", {})
    input_directory = directories.get(
        "data_input_directory_no_embedding",
        directories.get("data_input_directory"),
    )
    if input_directory is None:
        raise KeyError(
            "Missing directories.data_input_directory_no_embedding and "
            f"directories.data_input_directory in {config_path}. One of "
            "these paths is required for BuildDatasetNoEmbedding."
        )
    in_dir = Path(input_directory)

    label_map = {
        "Wjets": 1,
        "data": 2,
    }

    dfs = []
    missing_required = [
        str(in_dir / f"{name}.root")
        for name in REQUIRED_TRUE_TAU_PROCESSES
        if not (in_dir / f"{name}.root").is_file()
    ]
    if missing_required:
        missing = "\n".join(f"  - {path}" for path in missing_required)
        raise FileNotFoundError(
            "Missing true-tau input ROOT files required for "
            f"DR_qcd_extrapolation:\n{missing}"
        )

    process_counts = {}
    for name, process_id in PROCESSES.items():
        path = in_dir / f"{name}.root"

        df = uproot.open(path)["ntuple"].arrays(library="pd").copy()
        df = _ensure_event_column(df, process_id, name)

        df["process"] = process_id
        df["Label"] = label_map.get(name, 0)
        process_counts[name] = len(df)

        dfs.append(df)

    empty_required = [
        name for name in REQUIRED_TRUE_TAU_PROCESSES
        if process_counts.get(name, 0) == 0
    ]
    if empty_required:
        raise RuntimeError(
            "True-tau input files were found but contain zero events: "
            f"{empty_required}"
        )

    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # features
    df["SS"] = df["q_1"] * df["q_2"] > 0
    df["OS"] = df["q_1"] * df["q_2"] < 0
    df["parity"] = df["event"] % 2

    njets = df["njets"].clip(upper=2)

    df["class_weights"] = get_class_weights(
        weights=df["weight"],
        Y=njets,
        classes=(0, 1),
        class_weighted=False,
    )

    return df
