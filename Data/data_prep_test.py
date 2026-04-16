import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tap import Tap
from typing import Literal, Union
import torch as t
import uproot

dataset_names = ['data', 'diboson', 'dyjets', 'embedding', 'singletop', 'ttbar', 'wjets']
datasets = [0] * len(dataset_names) # initialize list to store dataframes

class Args(Tap):
    loc: Literal["remote", "present"] = "present"
    test: bool = False

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

    if args.test:
        if args.loc == "remote":
            file = "/run/user/1003/gvfs/sftp:host=portal1.etp.kit.edu,user=tapp//work/tapp/TauFF/NF4FF/Data/test_data/out_test_2018_data_tt.root"
        else:
            file = "/work/tapp/TauFF/NF4FF/Data/test_data/out_test_2018_data_tt.root"
        data = load_root_file_as_pd(file)
        data['SS'] = (data['q_1'] * data['q_2']) > 0 #change 1 * charge 2, if same sign: >0
        print(list(data.columns))
        print(data['njets'])
        #print(data[(data.trg_single_tau180_2)])
        #print(3>0)
        exit()
    


    for i in range(len(datasets)):

        if args.loc == "present":
            file = f"/work/tapp/TauFF/NF4FF/Data/test_data/out_test_2018_{dataset_names[i]}_tt.root"
        elif args.loc == "remote":
            file = f"/run/user/1003/gvfs/sftp:host=portal1.etp.kit.edu,user=tapp/work/tapp/TauFF/NF4FF/Data/test_data/out_test_2018_{datasets[i]}_tt.root"
        else:
            print("Invalid location specified. Use 'remote' or 'present'.")
            exit()
        
        # ----- Load the ROOT file and convert it to a pandas DataFrame -----
        datasets[i] = load_root_file_as_pd(file)

        
        # ----- Add additional columns to the DataFrame -----

        # set process name
        datasets[i]['process']= i
        datasets[i]['weight']= 1.0 #Todo: find out how to get the correct weights, currently set to 1 for all
        
        # set njets to 2 if it is greater than or equal to 2
        #datasets[i]['njets'].values[datasets[i]['njets'].values >= 2] = 2
        njets = datasets[i]['njets'].copy()
        njets[njets >= 2] = 2
        datasets[i]['class_weights'] = 1.0 #Todo: get_class_weights(weights = datasets[i].weight, Y = datasets[i].njets, classes = (0, 1, 2), class_weighted=False)
        
        # same sign, opposite sign
        datasets[i]['SS'] = (datasets[i]['q_1'] * datasets[i]['q_2']) > 0 #change 1 * charge 2, if same sign: >0
        datasets[i]['OS'] = (datasets[i]['q_1'] * datasets[i]['q_2']) < 0
        
        if dataset_names[i] == 'data':
            datasets[i]['Label'] = 2
        elif dataset_names[i] == 'wjets':
            datasets[i]['Label'] = 1
        else:
            datasets[i]['Label'] = 0
        
        if i == 0:
            print('works')
        

    combined_data = pd.concat(datasets, ignore_index=True)

    if args.loc == "present":
        combined_data.to_feather("/work/tapp/TauFF/NF4FF/Data/datasets/combined_data_test.feather")
    elif args.loc == "remote":
        combined_data.to_feather("/run/user/1003/gvfs/sftp:host=portal1.etp.kit.edu,user=tapp/work/tapp/TauFF/NF4FF/Data/datasets/combined_data_test.feather")
    else:
        print("Invalid location specified. Use 'remote' or 'present'.")
        exit()

if __name__ == "__main__":
    main()