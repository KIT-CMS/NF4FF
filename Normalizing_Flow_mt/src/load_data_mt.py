from pathlib import Path
import awkward as ak
import yaml
import numpy as np
import logging
from CustomLogging import setup_logging
import uproot
import pandas as pd
logger = setup_logging(logger=logging.getLogger(__name__))

# ----- define paths -----

fname = "/ceph/mmoser/FFmethod/smhtt_ul/2018_mt_2026-01-10/preselection/2018/mt/Wjets.root"

masks_path = "../configs/masks.yaml"
out_dir = '../data/'



# ----- classes and definitions -----

class MaskLoader:
    def __init__(
        self,
        config_path,
        mask_names,
        root_tree,
    ):
        self.config_path = Path(config_path)
        self.mask_names = mask_names if isinstance(mask_names, (list, tuple)) else [mask_names]
        self.root_dict = root_tree

        self.masks_dict = self._load_masks()
        self.masks = self._evaluate_masks()

    @staticmethod
    def convert_operators(expr: str) -> str:
        return (
            expr.replace("&&", "&")
                .replace("||", "|")
                .replace("!", "~")
        )

    def _load_masks(self):
        """Load all mask expressions from YAML for the requested mask names."""
        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        masks = {}
        for name in self.mask_names:
            group = data.get(name, {})
            if not group:
                raise KeyError(f"No masks found for '{name}'")
            masks.update(group)

        return masks

    def _evaluate_masks(self):
        """Evaluate all mask expressions into boolean numpy arrays."""
        evaluated = []

        for name, expr in self.masks_dict.items():
            expr = self.convert_operators(expr)
            mask = eval(expr, {}, self.root_dict)

            if isinstance(mask, tuple) and len(mask) == 1:
                mask = mask[0]
            
            if isinstance(mask, ak.Array):
                mask = ak.to_numpy(mask)

            if hasattr(mask, "numpy"):
                mask = mask.numpy()

            evaluated.append(mask.astype(bool))

        return evaluated

    def combine(self, mode="and"):
        """
        Combine masks using logical AND / OR.

        mode: 'and' or 'or'
        """
        if not self.masks:
            raise ValueError("No masks to combine")

        if mode == "and":
            return np.logical_and.reduce(self.masks)
        elif mode == "or":
            return np.logical_or.reduce(self.masks)
        else:
            raise ValueError("mode must be 'and' or 'or'")

    def apply(self, array, mode="and"):
        """Apply combined mask to an array."""
        mask = self.combine(mode=mode)
        return np.asarray(array)[mask]

def create_data_file(
    events,
    masks,
    name,
    config_path="../configs/masks.yaml",
    out_dir="../data/",
):
    # --- build & apply masks
    loader = MaskLoader(
        config_path=config_path,
        mask_names=masks,
        root_tree=events,
    )
    masked = loader.apply(events, mode="and")

    # --- construct dataframe
    df = pd.DataFrame({
        "pt_1": masked["pt_1"],
        "pt_2": masked["pt_2"],
        "m_vis": masked["m_vis"],
        "deltaR": masked["deltaR_ditaupair"],
        "njets": masked["njets"],
        "weight": masked["weight"],
    })

    # --- save
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_dir / f"{name}.pkl")

    return df

# --------------

    # --- event loading ---

file = uproot.open(fname)

events = file['ntuple;5'].arrays()

regions = {
    "SR_like_tight": ["event_preselection", "SR_like"],
    "SR_like_loose": ["event_preselection_loose", "SR_like"],
    "AR_like_tight": ["event_preselection", "AR_like"],
    "AR_like_loose": ["event_preselection_loose", "AR_like"],

    "SR_tight": ["event_preselection", "SR"],
    "SR_loose": ["event_preselection_loose", "SR"],
    "AR_tight": ["event_preselection", "AR"],
    "AR_loose": ["event_preselection_loose", "AR"]
}

for name, masks in regions.items():
    create_data_file(events, masks, name, config_path=masks_path, out_dir=out_dir)