import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tap import Tap
from typing import Literal, Union, Dict, Any
import torch as t
import uproot
import yaml

def load_root_file_as_pd(file_path):
    with uproot.open(file_path) as file:
        data = file["ntuple"].arrays(file["ntuple"].keys(), library="pd")
    return data

df_data = load_root_file_as_pd('/work/tapp/TauFF/NF4FF/Data/test_data/out_test_fastmtt_data_tt.root')
df_embed = load_root_file_as_pd('/work/tapp/TauFF/NF4FF/Data/test_data/out_test_fastmtt_embedding_tt.root')
df_classic = load_root_file_as_pd('/ceph/jvoss/FFmethod/smhtt_ul_v12/preselection/2018/tt/data.root')#/ceph/sgiappic/CMS_FF/260602/preselection/2024/tt/data.root')
print(list(df_classic.columns))

plt.hist(df_classic['m_fastmtt'], bins=50)
plt.savefig('/work/tapp/TauFF/NF4FF/Data/test_data/test.png')


