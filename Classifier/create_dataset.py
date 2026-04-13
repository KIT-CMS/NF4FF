from typing import Union

import numpy as np
import pandas as pd
import torch as t
import uproot
import logging
import yaml
from CustomLogging import setup_logging, LogContext
from pathlib import Path
from classes.helper import get_class_weights
# --------------------

SEED = 42
RANDOM_STATE = 42

# ------ logger -----

logger = setup_logging(logger=logging.getLogger(__name__))

# -------------------

config_path = Path(__file__).parent / "configs/config_settings.yaml"    
with config_path.open('r') as f:
    cfg = yaml.safe_load(f)

in_dir = cfg['directories']['data_input_directory']
out_dir = cfg['directories']['data_output_directory']

# --------------

def get_class_weights(
    weights: Union[pd.Series, np.ndarray, t.Tensor],
    Y: Union[pd.Series, np.ndarray, t.Tensor],
    classes: tuple = (0, 1),
    class_weighted: bool = True,
) -> Union[pd.Series, np.ndarray, t.Tensor]:
    _weights = np.zeros_like(weights)
    for _class in classes:
        _weights[Y == _class] = weights.sum() / weights[Y == _class].sum()
    return _weights * (weights if class_weighted else 1.0)

processes = {
    'Wjets': uproot.open(str(in_dir + "/" + "Wjets.root"))['ntuple'].arrays(library="pd"),
    'data': uproot.open(str(in_dir + "/" + "data.root"))['ntuple'].arrays(library="pd"),
    'diboson_J': uproot.open(str(in_dir + "/" + "diboson_J.root"))['ntuple'].arrays(library="pd"),
    'diboson_L': uproot.open(str(in_dir + "/" + "diboson_L.root"))['ntuple'].arrays(library="pd"),
    'DYjets_J': uproot.open(str(in_dir + "/" + "DYjets_J.root"))['ntuple'].arrays(library="pd"),
    'DYjets_L': uproot.open(str(in_dir + "/" + "DYjets_L.root"))['ntuple'].arrays(library="pd"),
    'ST_J': uproot.open(str(in_dir + "/" + "ST_J.root"))['ntuple'].arrays(library="pd"),
    'ST_L': uproot.open(str(in_dir + "/" + "ST_L.root"))['ntuple'].arrays(library="pd"),
    'ttbar_J': uproot.open(str(in_dir + "/" + "ttbar_J.root"))['ntuple'].arrays(library="pd"),
    'ttbar_L': uproot.open(str(in_dir + "/" + "ttbar_L.root"))['ntuple'].arrays(library="pd"),
    'embedding': uproot.open(str(in_dir + "/" + "embedding.root"))['ntuple'].arrays(library="pd")
}

dfs = []

for name, df in processes.items():
    df = df.copy()
    
    logger.info(f"Processing {name} with {len(df)} entries.")

    if name == 'data':
        df["process"] = 0
    elif name == 'Wjets':
        df['process'] = 1
    elif name == 'diboson_J':
        df['process'] = 2
    elif name == 'diboson_L':
        df['process'] = 3
    elif name == 'DYjets_J':
        df['process'] = 4
    elif name == 'DYjets_L':
        df['process'] = 5
    elif name == 'ST_J':
        df['process'] = 6
    elif name == 'ST_L':
        df['process'] = 7
    elif name == 'ttbar_J':
        df['process'] = 8
    elif name == 'ttbar_L':
        df['process'] = 9
    elif name == 'embedding':
        df['process'] = 10

    if name == 'Wjets':
        df["Label"] = 1
    elif name == 'data':
        df['Label'] = 2
    else:
        df['Label'] = 0
    
    dfs.append(df)

df = pd.concat(dfs, axis = 0, ignore_index=True)
df = df.sample(frac=1, random_state = RANDOM_STATE).reset_index(drop=True)

mask_SS =  (df.q_1 * df.q_2 > 0) 
mask_OS =  (df.q_1 * df.q_2 < 0) 

logger.info('Add event variable for SS/OS classification.')

df['SS'] = mask_SS
df['OS']= mask_OS

df['event_var'] = df['event']%2

njets = df.njets.copy()
njets[njets >= 2] = 2
df['class_weights'] = get_class_weights(
    weights = df.weight,
    Y = njets,
    classes = (0, 1, 2),
    class_weighted=False
)

df.to_feather(str(Path(out_dir) / "data_complete.feather"))
logger.info(f"Saved complete dataset to {str(Path(out_dir) / 'data_complete.feather')}.")