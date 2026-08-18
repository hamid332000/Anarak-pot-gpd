import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto


def plot_tail_probability(
    exceedances,
    sigma,
    xi,
    outfile
):
    """
    Empirical survival probability versus
    fitted GPD survival probability.
    """

    x = np.sort(exceedances)

    n = len(x)

    # -----------------------------------------
    # Empirical survival probability
    # -----------------------------------------

    empirical = (n - np.arange(1, n + 1) + 1) / (n + 1)

    # -----------------------------------------
    # Fitted GPD survival probability
    # -----------------------------------------

    theoretical = genpareto.sf(
        x,
        c=xi,
        loc=0,
        scale=sigma
    )

    fig, ax = plt.subplots(
        figsize=(6.5,6)
    )

    ax.scatter(
        theoretical,
        empirical,
        s=35,
        edgecolor="black",
        linewidth=0.5,
        zorder=3
    )

    # 1:1 reference line

    lim = [
        min(theoretical.min(), empirical.min()),
        1.0
    ]

    ax.plot(
        lim,
        lim,
        "r--",
        lw=2
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(lim)
    ax.set_ylim(lim)

    ax.set_xlabel(
        r"Theoretical survival probability $1-F(x)$",
        fontsize=12
    )

    ax.set_ylabel(
        r"Empirical survival probability $1-\hat{F}(x)$",
        fontsize=12
    )

    ax.grid(
        which="both",
        alpha=0.30
    )

    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()