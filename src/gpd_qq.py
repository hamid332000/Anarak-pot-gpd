import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

import config


def plot_gpd_qq(
    exceedances,
    sigma,
    xi,
    outfile,
    B=500
):
    """
    Publication-quality GPD QQ plot with
    95% simulation envelope.
    """

    x = np.sort(exceedances)

    n = len(x)

    p = (np.arange(1, n + 1) - 0.5) / n

    theoretical = genpareto.ppf(
        p,
        c=xi,
        loc=0,
        scale=sigma
    )

    # -----------------------------------------------
    # Bootstrap simulation envelope
    # -----------------------------------------------

   

    simulated = np.zeros((B, n))

    for b in range(B):

        sample = genpareto.rvs(
            c=xi,
            scale=sigma,
            size=n
        )

        simulated[b, :] = np.sort(sample)

    lower = np.percentile(
        simulated,
        2.5,
        axis=0
    )

    upper = np.percentile(
        simulated,
        97.5,
        axis=0
    )

    # -----------------------------------------------
    # Figure
    # -----------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6.5, 6)
    )

    ax.fill_between(
        theoretical,
        lower,
        upper,
        color="lightgray",
        alpha=0.60,
        label="95% simulation envelope"
    )

    ax.plot(
        theoretical,
        theoretical,
        "r--",
        lw=2,
        label="1:1 line"
    )

    ax.scatter(
        theoretical,
        x,
        s=35,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
        label="Observed excesses"
    )

    ax.set_xlabel(
    "Theoretical excess (°C)"
    )

    ax.set_ylabel(
    "Observed excess (°C)"
    )

    ax.grid(
        alpha=0.30
    )

    ax.legend()

    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()