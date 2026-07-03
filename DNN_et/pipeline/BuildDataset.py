from pathlib import Path
import pandas as pd
import uproot
import yaml

from classes import get_class_weights


def build_dataset(config_path: str) -> pd.DataFrame:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    in_dir = Path(cfg["directories"]["data_input_directory"])

    processes = {
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
        "embedding": 10,
    }

    label_map = {
        "Wjets": 1,
        "data": 2,
    }

    dfs = []
    for name, process_id in processes.items():
        path = in_dir / f"{name}.root"

        df = uproot.open(path)["ntuple"].arrays(library="pd").copy()

        df["process"] = process_id
        df["Label"] = label_map.get(name, 0)

        dfs.append(df)

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
