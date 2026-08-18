"""
Plots for dependence diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.dependence import autocorrelation


def plot_acf(
    series,
    nlags=10,
    title="Autocorrelation of declustered exceedances",
    outfile=None
):
    """
    Plot the sample autocorrelation function (ACF).

    Parameters
    ----------
    series : array-like
        Sequence of declustered exceedances.

    nlags : int, default=10
        Maximum lag displayed.

    title : str
        Figure title.

    outfile : pathlib.Path or None, default=None
        Output filename. If None, the figure is displayed.
    """

    values = autocorrelation(
        series,
        nlags
    )

    # Remove lag 0 (always equal to 1)
    lags = np.arange(1, nlags + 1)
    values = values[1:]

    # Approximate 95% confidence limits
    conf = 1.96 / np.sqrt(len(series))

    plt.figure(figsize=(7, 4))

    plt.stem(
        lags,
        values
    )

    plt.axhline(
        0,
        color="black",
        linewidth=1
    )

    plt.axhline(
        conf,
        color="red",
        linestyle="--",
        linewidth=1,
        label="95% confidence limits"
    )

    plt.axhline(
        -conf,
        color="red",
        linestyle="--",
        linewidth=1
    )

    plt.xlim(0.5, nlags + 0.5)

    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title(title)

    plt.legend()

    plt.tight_layout()

    if outfile is None:

        plt.show()

    else:

        plt.savefig(
            outfile,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()