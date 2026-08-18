"""
Mean Residual Life (MRL) plot with bootstrap confidence intervals.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_mrl(
    data,
    thresholds,
    selected=None,
    stable_min=None,
    stable_max=None,
    outfile=None,
    B=2000,
    ):
    
  
    """
    Mean Residual Life plot with bootstrap confidence intervals.

    Parameters
    ----------
    data : ndarray
        Original observations.
    thresholds : ndarray
        Candidate thresholds.
    selected : float or None
        Selected threshold.
    stable_min : float or None
        Lower bound of stable region.
    stable_max : float or None
        Upper bound of stable region.
    outfile : Path or str
        Output figure filename.
    B : int
        Number of bootstrap samples.
    """

    rng = np.random.default_rng()

    x = []
    mean_excess = []
    lower = []
    upper = []

    for u in thresholds:

        excess = data[data > u] - u

        if len(excess) < 5:
            continue

        x.append(u)

        mean_excess.append(np.mean(excess))

        n = len(excess)

        boot = np.empty(B)

        for b in range(B):
            sample = rng.choice(
                excess,
                size=n,
                replace=True,
            )

            boot[b] = np.mean(sample)

        lower.append(
            np.percentile(boot, 2.5)
        )

        upper.append(
            np.percentile(boot, 97.5)
        )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.fill_between(
        x,
        lower,
        upper,
        color="lightgray",
        alpha=0.40,
        label="95% bootstrap CI",
    )

    ax.plot(
        x,
        mean_excess,
        "o-",
        lw=2,
        ms=5,
        label="Mean excess",
    )

    # Optional vertical reference lines


    if selected is not None:
        ax.axvline(
            selected,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Selected threshold",
        )

    ax.set_xlabel(
        r"Threshold ($^\circ$C)"
    )

    ax.set_ylabel(
        r"Mean excess ($^\circ$C)"
    )

    ax.set_title(
        "Mean Residual Life Plot"
    )

    ax.grid(
        alpha=0.30
    )

    ax.legend()

    if outfile is not None:
        fig.savefig(
            outfile,
            dpi=600,
            bbox_inches="tight",
        )

    plt.close(fig)
    
    