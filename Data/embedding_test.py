import numpy as np
import matplotlib.pyplot as plt
from tap import Tap
from typing import Literal, Union, Dict, Any
import torch as t
import uproot
import yaml
import pandas as pd

def load_root_file_as_pd(file_path):
    with uproot.open(file_path) as file:
        data = file["ntuple"].arrays(file["ntuple"].keys(), library="pd")
    return data

def preselec_mask(df):
    had_tau_decay_mode_1 = (df.tau_decaymode_1 == 0) | (df.tau_decaymode_1 == 1) | (df.tau_decaymode_1 == 10) | (df.tau_decaymode_1 == 11)
    had_tau_decay_mode_2 = (df.tau_decaymode_2 == 0) | (df.tau_decaymode_2 == 1) | (df.tau_decaymode_2 == 10) | (df.tau_decaymode_2 == 11)
    had_tau_id_vs_ele = (df.id_tau_vsEle_VVLoose_1 > 0.5) & (df.id_tau_vsEle_VVLoose_2 > 0.5)
    had_tau_id_vs_mu = (df.id_tau_vsMu_VLoose_1 > 0.5) & (df.id_tau_vsMu_VLoose_2 > 0.5)
    had_tau_pt = (df.pt_1 > 40) & (df.pt_2 > 40)
    double_trigger = (df.trg_double_tau35_tightiso_tightid > 0.5) | (df.trg_double_tau35_mediumiso_hps > 0.5) | (df.trg_double_tau40_mediumiso_tightid > 0.5) | (df.trg_double_tau40_tightiso > 0.5)

    mask = had_tau_decay_mode_1 & had_tau_decay_mode_2 & had_tau_id_vs_ele & had_tau_id_vs_mu & had_tau_pt & double_trigger
    return mask

path = "/work/tapp/crown/KingMaker/CROWN/build/bin/"
file = "test_embed_tt.root"

df = load_root_file_as_pd(path + file)
print('----- before preselec -----')
print(len(df))

print('----- after preselec -----')
mask = preselec_mask(df)
print(len(df[mask]))

had_tau_decay_mode_1 = (df.tau_decaymode_1 == 0) | (df.tau_decaymode_1 == 1) | (df.tau_decaymode_1 == 10) | (df.tau_decaymode_1 == 11)
had_tau_decay_mode_2 = (df.tau_decaymode_2 == 0) | (df.tau_decaymode_2 == 1) | (df.tau_decaymode_2 == 10) | (df.tau_decaymode_2 == 11)
had_tau_id_vs_ele = (df.id_tau_vsEle_VVLoose_1 > 0.5) & (df.id_tau_vsEle_VVLoose_2 > 0.5)
had_tau_id_vs_mu = (df.id_tau_vsMu_VLoose_1 > 0.5) & (df.id_tau_vsMu_VLoose_2 > 0.5)
had_tau_pt = (df.pt_1 > 40) & (df.pt_2 > 40)
double_trigger = (df.trg_double_tau35_tightiso_tightid > 0.5) | (df.trg_double_tau35_mediumiso_hps > 0.5) | (df.trg_double_tau40_mediumiso_tightid > 0.5) | (df.trg_double_tau40_tightiso > 0.5)


print('----- after tau pt -----')
print(len(df[had_tau_pt]))
print('----- after double trigger -----')
print(len(df[double_trigger]))

print('----- after tau35 tight iso -----')
print(len(df[(df.trg_double_tau35_tightiso_tightid > 0.5)]))
print('----- after tau35 medium iso -----')
print(len(df[(df.trg_double_tau35_mediumiso_hps > 0.5)]))
print('----- after tau40 medium iso -----')
print(len(df[(df.trg_double_tau40_mediumiso_tightid > 0.5)]))
print('----- after tau40 tight iso -----')
print(len(df[(df.trg_double_tau40_tightiso > 0.5)]))