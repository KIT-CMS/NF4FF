import itertools as itt
import uproot
import logging
import pathlib
import pickle
import random
from collections import defaultdict
from dataclasses import KW_ONLY, dataclass, is_dataclass
from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Protocol, Tuple,
                    Union, runtime_checkable)

import numpy as np
import pandas as pd
import torch as t
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, scale
from tqdm import tqdm
from contextlib import contextmanager
from typing import (Any, Callable, Dict, Generator, Iterable, Iterator, List,
                    Optional, Tuple, Type, Union, get_args, get_origin)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold1, fold2 = train_test_split(
        df, test_size=0.5, random_state=SEED
    )
    train1, val1 = train_test_split(
        fold1, test_size=0.5, random_state=SEED
    )
    train2, val2 = train_test_split(
        fold2, test_size=0.5, random_state=SEED
    )

    return train1.reset_index(drop=True), val1.reset_index(drop=True), train2.reset_index(drop=True), val2.reset_index(drop=True)

# ----- seed -----

SEED = 42
RANDOM_STATE = 42

fname = "/ceph/mmoser/FFmethod/smhtt_ul/2018_et_2026-02-04/preselection/2018/et/"
out_dir = '../data/MC_data/'

# -----

processes = {
    'Wjets': uproot.open(str(fname + "/" + "Wjets.root"))['ntuple'].arrays(library="pd"),
    'data': uproot.open(str(fname + "/" + "data.root"))['ntuple'].arrays(library="pd"),
    'diboson_J': uproot.open(str(fname + "/" + "diboson_J.root"))['ntuple'].arrays(library="pd"),
    #'diboson_T': uproot.open(str(fname + "/" + "diboson_T.root"))['ntuple'].arrays(library="pd"),
    'diboson_L': uproot.open(str(fname + "/" + "diboson_L.root"))['ntuple'].arrays(library="pd"),
    'DYjets_J': uproot.open(str(fname + "/" + "DYjets_J.root"))['ntuple'].arrays(library="pd"),
    #'DYjets_T': uproot.open(str(fname + "/" + "DYjets_T.root"))['ntuple'].arrays(library="pd"),
    'DYjets_L': uproot.open(str(fname + "/" + "DYjets_L.root"))['ntuple'].arrays(library="pd"),
    'ST_J': uproot.open(str(fname + "/" + "ST_J.root"))['ntuple'].arrays(library="pd"),
    #'ST_T': uproot.open(str(fname + "/" + "ST_T.root"))['ntuple'].arrays(library="pd"),
    'ST_L': uproot.open(str(fname + "/" + "ST_L.root"))['ntuple'].arrays(library="pd"),
    'ttbar_J': uproot.open(str(fname + "/" + "ttbar_J.root"))['ntuple'].arrays(library="pd"),
    #'ttbar_T': uproot.open(str(fname + "/" + "ttbar_T.root"))['ntuple'].arrays(library="pd"),
    'ttbar_L': uproot.open(str(fname + "/" + "ttbar_L.root"))['ntuple'].arrays(library="pd"),
    'embedding': uproot.open(str(fname + "/" + "embedding.root"))['ntuple'].arrays(library="pd")
}

dfs = []

for name, df in processes.items():
    df = df.copy()
    for i, p in enumerate(processes):
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

#mask_SS = (df.pt_1 < 70) & (df.nbtag >= 0) & (df.q_1 * df.q_2 > 0) & (df.iso_1 > 0.0) & (df.iso_1 < 0.15)
#mask_OS = (df.pt_1 < 70) & (df.nbtag >= 0) & (df.q_1 * df.q_2 < 0) & (df.iso_1 > 0.0) & (df.iso_1 < 0.15)

mask_SS =  (df.q_1 * df.q_2 > 0) 
mask_OS =  (df.q_1 * df.q_2 < 0) 

df['SS'] = mask_SS
df['OS']= mask_OS

# SR-like masks

mask_s1 = (df.id_tau_vsJet_Tight_2 > 0.5)
mask_s2 = (df.nbtag == 0)
mask_s3 = ((df.q_1 * df.q_2) < 0)
mask_s4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
mask_s5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )

mask_SR_like = (mask_s1 & mask_s2 & mask_s3 & mask_s4 & mask_s5)

# AR-like mask

mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5) & (df.id_tau_vsJet_Tight_2 < 0.5))
mask_a2 = (df.nbtag == 0)
mask_a3 = ((df.q_1 * df.q_2) < 0)
mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )

mask_AR_like = (mask_a1 & mask_a2 & mask_a3 & mask_a4 & mask_a5)


df.to_feather('../data/data_complete.feather')

df_SR_like = df[mask_SR_like]
df_AR_like = df[mask_AR_like]

df_AR_like.to_feather('../data/data_AR_like.feather')
df_SR_like.to_feather('../data/data_SR_like.feather')


train1, val1, train2, val2 = split_data(df)

train1.to_feather('../data/train1.feather')
val1.to_feather('../data/val1.feather')
train2.to_feather('../data/train2.feather')
val2.to_feather('../data/val2.feather')