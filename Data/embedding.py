import numpy as np
import matplotlib.pyplot as plt
from tap import Tap
from typing import Literal, Union, Dict, Any
import torch as t
import uproot
import yaml
import pandas as pd


class Args(Tap):
    loc: Literal["remote", "present"] = "remote"

# ----- functions to load files -----

def load_root_file_as_pd(file_path):
    with uproot.open(file_path) as file:
        data = file["ntuple"].arrays(file["ntuple"].keys(), library="pd")
    return data

def preselec_mask(data_total):
    had_tau_decay_mode_1 = (data_total.tau_decaymode_1 == 0) | (data_total.tau_decaymode_1 == 1) | (data_total.tau_decaymode_1 == 10) | (data_total.tau_decaymode_1 == 11)
    had_tau_decay_mode_2 = (data_total.tau_decaymode_2 == 0) | (data_total.tau_decaymode_2 == 1) | (data_total.tau_decaymode_2 == 10) | (data_total.tau_decaymode_2 == 11)
    had_tau_id_vs_ele = (data_total.id_tau_vsEle_VVLoose_1 > 0.5) & (data_total.id_tau_vsEle_VVLoose_2 > 0.5)
    had_tau_id_vs_mu = (data_total.id_tau_vsMu_VLoose_1 > 0.5) & (data_total.id_tau_vsMu_VLoose_2 > 0.5)
    had_tau_pt = (data_total.pt_1 > 40) & (data_total.pt_2 > 40)
    double_trigger = (data_total.trg_double_tau35_tightiso_tightid > 0.5) | (data_total.trg_double_tau35_mediumiso_hps > 0.5) | (data_total.trg_double_tau40_mediumiso_tightid > 0.5) | (data_total.trg_double_tau40_tightiso > 0.5)

    mask = had_tau_decay_mode_1 & had_tau_decay_mode_2 & had_tau_id_vs_ele & had_tau_id_vs_mu & had_tau_pt & double_trigger
    return mask


# ----- main code -----

args = Args().parse_args()

if args.loc == "present":
    path = "/ceph/tapp/CROWN/ntuples/all_together/CROWNRun/2018/"
    preselec = "/work/tapp/TauFF/ClassicFF/TauFakeFactors/configs/smhtt_ul/2018/preselection_tt.yaml"
elif args.loc == "remote":
    path = "/run/user/1003/gvfs/sftp:host=portal1.etp.kit.edu,user=tapp/ceph/tapp/CROWN/ntuples/all_together/CROWNRun/2018/"
    preselec = "/run/user/1003/gvfs/sftp:host=portal1.etp.kit.edu,user=tapp/work/tapp/TauFF/ClassicFF/TauFakeFactors/configs/smhtt_ul/2018/preselection_tt.yaml"

preselec = yaml.safe_load(open(preselec, "r"))
print(preselec["event_selection"])



print("------------------------------")
print("----- without embedding, only mc -----")

pres = "/ceph/tapp/TauFF/smhtt_ul/2018/2026_04_17_FF/preselection/2018/tt/"
rem = "/run/user/1003/gvfs/sftp:host=portal1.etp.kit.edu,user=tapp/ceph/tapp/TauFF/smhtt_ul/2018/2026_04_17_FF/preselection/2018/tt/"

diboson = load_root_file_as_pd(rem + "diboson_T.root")
DYjets = load_root_file_as_pd(rem + "DYjets_T.root")
singletop = load_root_file_as_pd(rem + "ST_T.root")
ttbar = load_root_file_as_pd(rem + "ttbar_T.root")

data_total_mc = pd.concat([diboson, DYjets, singletop, ttbar], ignore_index=True)
print(list(data_total_mc.columns))
print("number of events in data_total_mc:", len(data_total_mc))

# ----- embedding data -----
print("------------------------------")
print("----- embedding -----")

part = ["A", "B", "C", "D"]

data= []

for x in part:
    folder = path + f"TauEmbedding-TauTauFinalState_Run2018{x}-UL2018/tt"
    for i in range(0, 20):
        file = folder + f"/TauEmbedding-TauTauFinalState_Run2018{x}-UL2018_{i}.root"
        try:
            datax = load_root_file_as_pd(file)
            #print(f"Loaded {file} with {len(datax)} entries.")
            data.append(datax)
        except:
            #print(f"Not found: {file}")
            continue




print(len(data))

data_total = pd.concat(data, ignore_index=True)

print(list(data_total.columns))
print(len(data_total))


plt.hist(data_total["pt_1"], bins=100)
plt.show()



# ----- with fast preselection -----


mask = preselec_mask(data_total)
print(len(data_total[mask]))


# ----- slow selection -----
print("------------------------------")
print("----- step by step selection -----")

had_tau_decay_mode_1 = (data_total.tau_decaymode_1 == 0) | (data_total.tau_decaymode_1 == 1) | (data_total.tau_decaymode_1 == 10) | (data_total.tau_decaymode_1 == 11)
had_tau_decay_mode_2 = (data_total.tau_decaymode_2 == 0) | (data_total.tau_decaymode_2 == 1) | (data_total.tau_decaymode_2 == 10) | (data_total.tau_decaymode_2 == 11)
had_tau_id_vs_ele = (data_total.id_tau_vsEle_VVLoose_1 > 0.5) & (data_total.id_tau_vsEle_VVLoose_2 > 0.5)
had_tau_id_vs_mu = (data_total.id_tau_vsMu_VLoose_1 > 0.5) & (data_total.id_tau_vsMu_VLoose_2 > 0.5)
had_tau_pt = (data_total.pt_1 > 40) & (data_total.pt_2 > 40)
double_trigger = (data_total.trg_double_tau35_tightiso_tightid > 0.5) | (data_total.trg_double_tau35_mediumiso_hps > 0.5) | (data_total.trg_double_tau40_mediumiso_tightid > 0.5) | (data_total.trg_double_tau40_tightiso > 0.5)

print(len(data_total[had_tau_decay_mode_1]))
print(len(data_total[had_tau_decay_mode_2]))
print(len(data_total[had_tau_id_vs_ele]))
print(len(data_total[had_tau_id_vs_mu]))
print(len(data_total[had_tau_pt]))
print(len(data_total[double_trigger]))