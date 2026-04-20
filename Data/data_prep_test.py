import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tap import Tap
from typing import Literal, Union, Dict, Any
import torch as t
import uproot
import yaml

dataset_names = ['data', 'diboson', 'dyjets', 'embedding', 'singletop', 'ttbar', 'wjets']
datasets = [0] * len(dataset_names) # initialize list to store dataframes


class Args(Tap):
    loc: Literal["remote", "present"] = "present"
    embedding: Literal["embedding", "no_embedding"] = "no_embedding"
    test: bool = False

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_root_file_as_pd(file_path):
    with uproot.open(file_path) as file:
        # part of explanation: https://www.youtube.com/watch?v=W0slvpV2spw
        data = file["ntuple"].arrays(file["ntuple"].keys(), library="pd")
    return data

# ----- Todo: not yet in use and it doesn't work
def get_class_weights(weights: Union[pd.Series, np.ndarray, t.Tensor], Y: Union[pd.Series, np.ndarray, t.Tensor], classes: tuple = (0, 1), class_weighted: bool = True,) -> Union[pd.Series, np.ndarray, t.Tensor]:
    _weights = np.zeros_like(weights)
    for _class in classes:
        _weights[Y == _class] = weights.sum() / weights[Y == _class].sum()
    return _weights * (weights if class_weighted else 1.0)

def main():

    args = Args().parse_args()

    if args.loc == "present":
        cfg = load_config("/work/tapp/TauFF/NF4FF/Data/config_datasets.yaml")
    elif args.loc == "remote":
        cfg = load_config("/run/user/1003/gvfs/sftp:host=portal1.etp.kit.edu,user=tapp/work/tapp/TauFF/NF4FF/Data/config_datasets.yaml")

    # ----- Define datasets used and initialize list to store dataframes -----
    
    dataset_names = [cfg['data']]
    dataset_names.extend(cfg[args.embedding])
    datasets = [0] * len(dataset_names)

    for x in dataset_names:
        file = cfg['input_dir'][args.loc] + x
        data = load_root_file_as_pd(file)
        print(f"Length of {x}: {len(data)}")
    


    

if __name__ == "__main__":
    main()