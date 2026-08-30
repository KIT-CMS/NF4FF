import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tap import Tap
from typing import Literal, Union, Dict, Any
import torch as t
import uproot
import yaml
import logging

from Logging import setup_logging

logger = setup_logging(logger=logging.getLogger(__name__))


class Args(Tap):
    loc: Literal["remote", "present"] = "present"
    embedding: Literal["embedding", "no_embedding"] = "embedding"
    classic: bool = False


# ----- functions to load files -----

def load_root_file_as_pd(file_path):
    with uproot.open(file_path) as file:
        data = file["ntuple"].arrays(file["ntuple"].keys(), library="pd")
    return data

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def write_yaml_to_file(py_obj, filename):
    with open(f'{filename}.yaml', 'w',) as f :
        yaml.dump(py_obj, f, sort_keys=False) 
    logger.info(f"Y{py_obj} written to {filename}.yaml")

# ----- calculate class weights -----
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

def get_class_weight_per_process(df, num_classes, process, class_weighted=True):
    num_rows = len(df)
    num_rows_class = len(df[df['process'] == process])
    class_weight = num_rows / (num_classes * num_rows_class)

    return class_weight if class_weighted else 1.0


def main():

    args = Args().parse_args()

    # ----- Load config -----

    if args.loc == "present":
        cfg = load_config("/work/tapp/TauFF/NF4FF/Data/config_datasets.yaml")
    elif args.loc == "remote":
        cfg = load_config("/run/user/1003/gvfs/sftp:host=portal1.etp.kit.edu,user=tapp/work/tapp/TauFF/NF4FF/Data/config_datasets.yaml")
    else:
        logger.error("Invalid location specified. Use 'remote' or 'present'.")
        exit()

    # ----- Define datasets used and initialize list to store dataframes -----
    
    dataset_names = [cfg['data']]
    dataset_names.extend(cfg[args.embedding])
    datasets = [0] * len(dataset_names)
    process_num = list(range(len(dataset_names)))

    names = []
    for x in dataset_names:
        #names.append(x.strip(".root"))
        names.append(x.removesuffix(".root"))
    print(names)
    print(dataset_names)

    data_process = dict(zip(names, process_num))

    write_yaml_to_file(data_process, cfg["process_dir"][args.loc] + "/data_process_mapping_" + args.embedding + ".yaml")
    #exit()

    

    for i in range(len(datasets)):
        if args.classic:
            #input_dir_classic_sgiappic or ..._jvoss --> also change save file name!!!
            file = cfg['input_dir_classic_sgiappic'] + dataset_names[i]
        else:
            file = cfg['input_dir'][args.embedding][args.loc] + dataset_names[i]
        
        # ----- Load the ROOT file and convert it to a pandas DataFrame -----
        datasets[i] = load_root_file_as_pd(file)

        if datasets[i].empty:
            logger.warning(f"{dataset_names[i]} is empty.")
            continue

        
        # ----- Add additional columns to the DataFrame -----

        # set process name
        datasets[i]['process']= i

        if not args.classic:
            # even or odd number of events
            datasets[i]['event_var'] = datasets[i]['event'] % 2
            
            # set njets to 2 if it is greater than or equal to 2
            datasets[i]['njets'].values[datasets[i]['njets'].values >= 2] = 2
            
            # same sign, opposite sign
            datasets[i]['SS'] = (datasets[i]['q_1'] * datasets[i]['q_2']) > 0 #change 1 * charge 2, if same sign: >0
            datasets[i]['OS'] = (datasets[i]['q_1'] * datasets[i]['q_2']) < 0

        #class weights for process
        #datasets[i]['class_weights_process'] = get_class_weight_per_process(datasets[i], len(dataset_names), i)
        print(dataset_names[i])
        if dataset_names[i] == 'data.root': datasets[i]['Label'] = 1
        else: datasets[i]['Label'] = 0

        logger.info(f"{dataset_names[i]} with {len(datasets[i])} events loaded.")
        logger.info(f"{dataset_names[i]} with {np.sum(datasets[i].weight)}.")

    combined_data = pd.concat(datasets, ignore_index=True)

    # set class weights for process
    combined_data['class_weights'] = get_class_weights(weights = combined_data.weight, Y = combined_data.Label, classes = (0, 1), class_weighted=True)
    
    if args.classic:
        out_classic = Path(cfg['output_dir'][args.loc] + "classic")
        out_classic.mkdir(parents=True, exist_ok=True)
        combined_data.to_feather(out_classic / "combined_data_sgiappic.feather")
    else:
        logger.info(f"Saving combined data to {cfg['output_dir'][args.loc] + args.embedding}/combined_data.feather")
        combined_data.to_feather(cfg['output_dir'][args.loc] + args.embedding + "/combined_data.feather")

if __name__ == "__main__":
    main()