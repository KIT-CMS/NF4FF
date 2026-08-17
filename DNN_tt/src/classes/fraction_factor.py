import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def pt_mask(df):
    bin_edges = [40, 45, 50 , 55, 60, 65, 70, 75, 80, 90, 100, 120, 200]
    bins = np.zeros((len(bin_edges), len(bin_edges)))
    for i in range(len(bin_edges)):
        mask_pt2 = (bin_edges[i]<=df.pt_2) & (df.pt_2 < bin_edges[i+1])
        for j in range(len(bin_edges)):
            mask_pt1 = (bin_edges[j]<=df.pt_1) & (df.pt_1 < bin_edges[j+1])    
            bins[i][j] = df[mask_pt1 & mask_pt2] 
    print(np.identity(3))
    return bins

def fraction_in_bins(df):
    return None

df = pd.read_feather('/work/tapp/TauFF/NF4FF/Data/datasets/embedding/combined_data_updated.feather')
print(pt_mask(df))