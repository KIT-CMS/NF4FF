import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from classes.Plotting import add_cms_privatework_lumi_row
from classes.Collection import load_config
import mplhep as hep


cfg = load_config("/work/tapp/TauFF/NF4FF/Normalizing_Flow_tt/configs/labels_short.yaml")



def corr_matrix_nfsample_data(data, nf_sample, var, title, tag, dir):
    hep.style.use(hep.style.CMS)  # Use CMS style for all plots in this category

    cor_dir = dir / 'matrices'
    cor_dir.mkdir(parents=True, exist_ok=True)

    labels = [cfg["tt"][k] for k in var if k in cfg["tt"]]
    data = data[var]

    # 1. Compute correlation matrices
    corr_true = np.corrcoef(data, rowvar=False)
    corr_flow = np.corrcoef(nf_sample, rowvar=False)
    

    # 2. Calculate the differences
    diff_matrix = np.abs(corr_true - corr_flow)

    specs  = [
        (corr_true, "corr_true", "True data"),
        (corr_flow, "corr_flow", "NF sampled"),
        (diff_matrix, "diff_matrix", "True vs. NF")
    ]

    # 3. Plotting the matrices

    for name, save, subtitle in specs:

        plt.figure(figsize=(13, 10.4))
        #add_cms_privatework_lumi_row(None)
        if len(var) >= 8:
            if save == "diff_matrix":
                sns.heatmap(name, 
                        cmap="crest",
                        vmin=0, vmax=1,
                        xticklabels=labels, yticklabels=labels, 
                        annot=True, annot_kws={"fontsize":12},
                        linewidths=0.1, fmt=".3f")
            else:
                sns.heatmap(name, 
                        cmap="RdBu",
                        vmin=-1, vmax=1,
                        xticklabels=labels, yticklabels=labels, 
                        annot=True, annot_kws={"fontsize":12},
                        linewidths=0.1, fmt=".3f")
        else:
            if save == "diff_matrix":
                sns.heatmap(name, 
                        cmap="crest",
                        vmin=0, vmax=1,
                        xticklabels=labels, yticklabels=labels, 
                        annot=True, annot_kws={"fontsize":15},
                        linewidths=0.1, fmt=".3f")
            else:
                sns.heatmap(name, 
                        cmap="RdBu",
                        vmin=-1, vmax=1,
                        xticklabels=labels, yticklabels=labels, 
                        annot=True, annot_kws={"fontsize":15},
                        linewidths=0.1, fmt=".3f")
        
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True) # labels along the bottom edge are on
        plt.tick_params(
            axis='y',          # changes apply to the y-axis
            which='both',      
            left=False,      
            right=False)
        
        #plt.title(f"Correlation Matrix: {subtitle} for {title}")
        plt.title(r"$\tau_h\tau_h, \; \it{Private \; work\; (CMS \; data/simulation)}$", loc='left')
        plt.title("59.8 $\mathrm{fb}^{-1}$ (2018, 13 TeV)", loc='right')
        plt.tight_layout()
        plt.savefig(cor_dir / f"{save}_{tag}.png")
        plt.close()

    