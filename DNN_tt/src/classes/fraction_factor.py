import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def pt_mask(df):
    bin_edges = [40, 45, 50 , 55, 60, 65, 70, 75, 80, 90, 100, 120, 200]
    n_bins = len(bin_edges) -1

    masks = np.empty((n_bins, n_bins))
    for i in range(n_bins):
        if i == np.max(range(n_bins)):
            mask_pt2 = ((bin_edges[i] <= df["pt_2"]))
        else:
            mask_pt2 = ((bin_edges[i] <= df["pt_2"]) & (df["pt_2"] < bin_edges[i + 1]))        

        for j in range(n_bins):
            if j == np.max(range(n_bins)):
                mask_pt1 = ((bin_edges[j] <= df["pt_1"]))
            else:
                mask_pt1 = ((bin_edges[j] <= df["pt_1"]) & (df["pt_1"] < bin_edges[j + 1]))

            masks[i, j] = mask_pt1 & mask_pt2

    return masks

def fraction_in_bins(df):
    return None

df = pd.read_feather('/work/tapp/TauFF/NF4FF/Data/datasets/embedding/combined_data_updated.feather')
print(pt_mask(df))