import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def plot_regions(
    signal: np.ndarray,
    data: np.ndarray,
    weights_MC: np.ndarray,
    weights_signal: np.ndarray,
    weights_data: np.ndarray,
    log_scale: bool,
    filename: str,
):
    bins = np.linspace(0, 1, 11)

    hist_signal, edges  = np.histogram(signal, bins=bins, weights=weights_signal)
    hist_data, _   = np.histogram(data, bins=bins, weights = weights_data)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths  = edges[1:] - edges[:-1]

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        centers, hist_data, np.sqrt(hist_data),
        label="data", marker=".", linestyle="None", color="black"
    )

    plt.bar(
        centers, hist_signal, width=widths,
        label="non FF-process",
        color="tab:blue", alpha=0.6, edgecolor="black"
    )

    if log_scale:
        plt.yscale("log")

    plt.ylim(1e2, None)
    plt.xlabel("NN output")
    plt.ylabel("Counts")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ----------------------------

def main():

    # Load data

    df_data = pd.read_feather("../../data/data_complete.feather")
    df_signal = pd.read_feather("../../data/data_signal.feather")


    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# ----------------------------

if __name__ == "__main__":
    main()