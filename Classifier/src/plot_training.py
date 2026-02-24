import pandas as pd
import matplotlib.pyplot as plt

def plot_losses(pickle_path, save_path=None):
    """
    Load training_logs.pkl and plot train/val losses.

    Parameters
    ----------
    pickle_path : str
        Path to the training_logs.pkl file.
    save_path : str or None
        If given, saves the plot to this path.
    """

    # load
    
    df = pd.read_pickle(pickle_path)

    # sort just in case
    
    df = df.sort_values("epoch")

    # plot

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", lw=2)
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", lw=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

plot_losses('Categorizer_results/training/fold1/2026-02-03/0_13-51-54/training_logs.pkl', '../plots/training_fold1.png')
plot_losses('Categorizer_results/training/fold1/2026-02-03/0_13-51-54/training_logs.pkl', '../plots/training_fold1.pdf')
#plot_losses('Categorizer_results/pretraining/fold2/2026-02-02/0_21-22-53/training_logs.pkl', '../plots/pretraining_fold2.png')
#plot_losses('Categorizer_results/pretraining/fold2/2026-02-02/0_21-22-53/training_logs.pkl', '../plots/pretraining_fold2.pdf')