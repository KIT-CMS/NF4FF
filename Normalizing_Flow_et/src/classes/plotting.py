import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, Literal, Union

Number = Union[int, float]
import logging
from CustomLogging import setup_logging
logger = setup_logging(logger=logging.getLogger(__name__))

class MultiDimHist:
    """
    Create and plot histograms for each dimension of a d-dimensional dataset.

    Parameters
    ----------
    bins : int or sequence or str, optional
        Number of bins (int), or a custom bins specification per dimension.
        If str, must be one of {'auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt'}.
        Default: 'auto'
    density : bool, optional
        If True, normalize histograms to form a probability density.
        If False, show raw counts. Default: False.
    range : None or list of tuples, optional
        Per-dimension (min, max) ranges for the histograms. If None, computed from data.
        If provided, should be a list/tuple of length d.
    labels : None or list of str, optional
        Per-dimension labels (used in plot titles). If None, uses 'Dim i'.
    """

    def __init__(self, bins='auto', density=False, range=None, labels=None):
        self.bins = bins
        self.density = density
        self.range = range
        self.labels = labels

        # Filled after calling fit()
        self.hist_counts = None   # list of arrays: counts per dimension
        self.bin_edges = None     # list of arrays: bin edges per dimension
        self.data_shape = None    # (n_samples, d)
        self.fitted = False

    def _to_numpy(self, x):
        """Convert torch.Tensor or list to numpy array."""
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except ImportError:
            pass
        logger.info(f"shape of array {np.shape(x)}")
        return np.asarray(x)

    def fit(self, x):
        """
        Compute histograms for input data.

        Parameters
        ----------
        x : array-like, shape (n_samples, d)
            Input dataset.

        Returns
        -------
        self
        """
        x = self._to_numpy(x)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array of shape (n_samples, d); got shape {x.shape}")
        n, d = x.shape
        self.data_shape = (n, d)

        # Prepare containers
        self.hist_counts = []
        self.bin_edges = []

        # Validate/prepare per-dimension options
        if self.range is not None:
            if len(self.range) != d:
                raise ValueError("range must be a list/tuple of length d with (min, max) per dimension.")
        if self.labels is not None and len(self.labels) != d:
            raise ValueError("labels must be a list/tuple of length d.")

        # For each dimension, compute histogram
        for j in range(d):
            dim_data = x[:, j]

            # Select bins and range per dimension
            bins_j = self.bins
            range_j = None if self.range is None else self.range[j]

            counts, edges = np.histogram(dim_data, bins=bins_j, range=range_j, density=self.density)
            self.hist_counts.append(counts)
            self.bin_edges.append(edges)

        self.fitted = True
        return self

    def plot(self, figsize=(10, 6), cols=3, sharey=False, tight_layout=True, savepath=None):
        """
        Plot the histograms for each dimension in a grid.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default: (10, 6)
        cols : int, optional
            Number of columns in the subplot grid. Default: 3
        sharey : bool, optional
            Whether to share y-axis across subplots. Default: False
        tight_layout : bool, optional
            If True, call plt.tight_layout(). Default: True
        savepath : str or None, optional
            If provided, save the figure to this filepath.
        """
        if not self.fitted:
            raise RuntimeError("Call fit(x) before plot().")

        _, d = self.data_shape
        rows = int(np.ceil(d / cols))

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, sharey=sharey)
        axes = axes.flatten()

        # Plot each dimension
        for j in range(d):
            ax = axes[j]
            counts = self.hist_counts[j]
            edges = self.bin_edges[j]

            # Use bar plot for histogram bins
            widths = np.diff(edges)
            lefts = edges[:-1]
            ax.bar(lefts, counts, width=widths, align='edge', edgecolor='black', alpha=0.7)

            title = self.labels[j] if (self.labels and j < len(self.labels)) else f"Dim {j}"
            ax.set_title(f"Histogram – {title}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density" if self.density else "Count")
            ax.grid(True, linestyle='--', alpha=0.3)

        # Hide unused subplots (if d < rows*cols)
        for k in range(d, rows * cols):
            fig.delaxes(axes[k])

        if tight_layout:
            plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath, dpi=150)
        return fig




@dataclass
class HistogramRatioPlotter:
    n_bins: Optional[int] = 40
    bin_edges: Optional[np.ndarray] = None
    range: Optional[Tuple[Number, Number]] = None
    density: bool = False
    show_ratio_errors: bool = True
    ratio_reference: float = 1.0
    stacked: bool = False  # stacked top panel; ratio still computed per dataset
    colors: Tuple[str, str] = ("tab:blue", "tab:orange")
    labels: Tuple[str, str] = ("Dataset A", "Dataset B")
    title: str = "Overlaid Histograms (Shared Bins)"
    x_label: str = "Value"
    y_label: str = "Counts"
    ratio_label: str = "Ratio (A / B)"
    grid_alpha: float = 0.15
    figsize: Tuple[Number, Number] = (10, 7)
    height_ratios: Tuple[int, int] = (3, 1)
    y_min_ratio: Optional[float] = 0.0  # None = autoscale

    # ---------- internal helpers ----------
    def _compute_bins(self, data_a: np.ndarray, data_b: np.ndarray) -> np.ndarray:
        """Derive shared bin edges."""
        if self.bin_edges is not None:
            return np.asarray(self.bin_edges, dtype=float)

        if self.range is not None:
            dmin, dmax = self.range
        else:
            dmin = np.nanmin([np.min(data_a), np.min(data_b)])
            dmax = np.nanmax([np.max(data_a), np.max(data_b)])

        n_bins = 40 if self.n_bins is None else int(self.n_bins)
        return np.linspace(dmin, dmax, n_bins + 1)

    @staticmethod
    def _poisson_ratio_errors(a: np.ndarray, b: np.ndarray, ratio: np.ndarray) -> np.ndarray:
        """
        Poisson error propagation for ratio r = A / B:
            σ_r ≈ r * sqrt(1/A + 1/B)
        Only defined where A>0 and B>0 and ratio finite.
        """
        err = np.full_like(ratio, np.nan, dtype=float)
        mask = (a > 0) & (b > 0) & np.isfinite(ratio)
        err[mask] = ratio[mask] * np.sqrt(1.0 / a[mask] + 1.0 / b[mask])
        return err

    def _hist_counts(
        self,
        data: np.ndarray,
        bins: np.ndarray,
        weights: Optional[Sequence[Number]],
        density: bool,
    ) -> np.ndarray:
        counts, _ = np.histogram(data, bins=bins, weights=weights, density=density)
        return counts

    # ---------- public API ----------
    def compute(
        self,
        data_a: Sequence[Number],
        data_b: Sequence[Number],
        weights_a: Optional[Sequence[Number]] = None,
        weights_b: Optional[Sequence[Number]] = None,
        include_errors: bool = True,
    ) -> pd.DataFrame:
        """
        Compute per-bin counts (or densities), A/B ratio, and optional ratio errors.
        Returns a tidy DataFrame with left/right/center edges and metrics.
        """
        data_a = np.asarray(data_a, dtype=float)
        data_b = np.asarray(data_b, dtype=float)

        bins = self._compute_bins(data_a, data_b)
        centers = 0.5 * (bins[:-1] + bins[1:])

        counts_a = self._hist_counts(data_a, bins, weights_a, density=self.density)
        counts_b = self._hist_counts(data_b, bins, weights_b, density=self.density)

        ratio = np.divide(
            counts_a, counts_b,
            out=np.full_like(counts_a, np.nan, dtype=float),
            where=counts_b != 0
        )

        if include_errors and self.show_ratio_errors and not self.density:
            err_ratio = self._poisson_ratio_errors(counts_a, counts_b, ratio)
        else:
            err_ratio = np.full_like(ratio, np.nan, dtype=float)

        df = pd.DataFrame({
            "bin_left": bins[:-1],
            "bin_right": bins[1:],
            "bin_center": centers,
            "counts_a": counts_a,
            "counts_b": counts_b,
            "ratio_a_over_b": ratio,
            "ratio_err": err_ratio,
        })

        return df

    def plot(
        self,
        data_a: Sequence[Number],
        data_b: Sequence[Number],
        weights_a: Optional[Sequence[Number]] = None,
        weights_b: Optional[Sequence[Number]] = None,
        save_path: Optional[str] = None,
        save_dpi: int = 150,
        save_bbox: Union[Literal["tight"], None] = "tight",
        save_transparent: bool = False,
        return_fig: bool = False,
    ):
        """
        Create the overlaid histogram plot + ratio panel and optionally save.
        Returns (fig, (ax_top, ax_ratio)) if return_fig=True else None.
        """
        # Use compute() to get consistent arrays
        df = self.compute(
            data_a=data_a,
            data_b=data_b,
            weights_a=weights_a,
            weights_b=weights_b,
            include_errors=True,
        )

        bins = np.concatenate([df["bin_left"].values, [df["bin_right"].values[-1]]])
        centers = df["bin_center"].values
        counts_a = df["counts_a"].values
        counts_b = df["counts_b"].values
        ratio = df["ratio_a_over_b"].values
        err_ratio = df["ratio_err"].values

        # --- Plotting ---
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=self.height_ratios, hspace=0.25)
        ax_top = fig.add_subplot(gs[0])

        if self.stacked:
            ax_top.hist(
                [np.asarray(data_a, dtype=float), np.asarray(data_b, dtype=float)],
                bins=bins,
                weights=[weights_a, weights_b],
                density=self.density,
                stacked=True,
                color=list(self.colors),
                label=list(self.labels),
                alpha=0.7,
                edgecolor="none",
            )
        else:
            ax_top.hist(
                data_a, bins=bins, weights=weights_a, density=self.density,
                histtype="stepfilled", alpha=0.4, color=self.colors[0],
                label=self.labels[0], edgecolor="none"
            )
            ax_top.hist(
                data_b, bins=bins, weights=weights_b, density=self.density,
                histtype="stepfilled", alpha=0.4, color=self.colors[1],
                label=self.labels[1], edgecolor="none"
            )
            ax_top.hist(
                data_a, bins=bins, weights=weights_a, density=self.density,
                histtype="step", lw=1.6, color=self.colors[0]
            )
            ax_top.hist(
                data_b, bins=bins, weights=weights_b, density=self.density,
                histtype="step", lw=1.6, color=self.colors[1]
            )

        ax_top.set_ylabel("Density" if self.density else self.y_label)
        ax_top.set_title(self.title)
        ax_top.legend()
        ax_top.grid(True, alpha=self.grid_alpha)

        ax_ratio = fig.add_subplot(gs[1], sharex=ax_top)
        ax_ratio.plot(centers, ratio, marker="o", ms=4, color="k", lw=1, label="A / B")

        if self.show_ratio_errors and np.any(np.isfinite(err_ratio)):
            ax_ratio.errorbar(
                centers, ratio, yerr=err_ratio, fmt="none",
                ecolor="gray", elinewidth=1, capsize=2, alpha=0.9
            )

        # if self.ratio_reference is not None:
        #     ax_ratio.axhline(self.ratio_reference, color="red", linestyle="--", linewidth=1, alpha=0.8)

        ax_ratio.set_xlabel(self.x_label)
        ax_ratio.set_ylabel(self.ratio_label)
        ax_ratio.grid(True, alpha=self.grid_alpha)
        if self.y_min_ratio is not None:
                       ax_ratio.set_ylim(bottom=self.y_min_ratio)

        plt.setp(ax_top.get_xticklabels(), visible=False)

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=save_dpi, bbox_inches=save_bbox, transparent=save_transparent)

        if return_fig:
            return fig, (ax_top, ax_ratio)
        else:
            plt.show()
            plt.close(fig)
def plot_histograms(
    data_list,
    config_list,
    titles=None,
    figsize=(10, 8),
    save_path=None,
    dpi=300,
    show=True,
    **default_hist_kwargs,
):
    """
    Plot 4 histograms in a 2x2 grid with per-histogram configuration.

    Parameters
    ----------
    data_list : list of array-like
        Exactly 4 datasets.
    config_list : list of dict
        Exactly 4 config dictionaries. Each may contain:
            - x_min (float)
            - x_max (float)
            - binwidth (float)
            - bins (int or array-like)
            - hist_kwargs (dict)
    titles : list of str, optional
        Titles for each subplot.
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
        Example: "figures/histograms.png"
    dpi : int
        Resolution for saved figure.
    show : bool
        Whether to display the figure.
    **default_hist_kwargs :
        Default kwargs passed to all histograms.
    """

    assert len(data_list) == 4, "data_list must have exactly 4 datasets"
    assert len(config_list) == 4, "config_list must have exactly 4 configs"

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        data = np.asarray(data_list[i])
        cfg = config_list[i]

        x_min = cfg.get("x_min")
        x_max = cfg.get("x_max")
        binwidth = cfg.get("binwidth")
        bins = cfg.get("bins")
        hist_kwargs = cfg.get("hist_kwargs", {})

        # Filter data
        if x_min is not None:
            data = data[data >= x_min]
        if x_max is not None:
            data = data[data <= x_max]

        # Determine bins
        if bins is not None:
            hist_bins = bins
        elif binwidth is not None:
            xmin = x_min if x_min is not None else data.min()
            xmax = x_max if x_max is not None else data.max()
            hist_bins = np.arange(xmin, xmax + binwidth, binwidth)
        else:
            hist_bins = None

        final_hist_kwargs = {**default_hist_kwargs, **hist_kwargs}

        ax.hist(data, bins=hist_bins, **final_hist_kwargs)

        if titles:
            ax.set_title(titles[i])

        if x_min is not None or x_max is not None:
            ax.set_xlim(x_min, x_max)

    plt.tight_layout()

    # Save figure if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

