import numpy as np
from scipy.stats import genpareto


def ad_statistic(data, sigma, xi):
    """
    Anderson–Darling statistic for fitted GPD.
    """

    x = np.sort(data)

    n = len(x)

    F = genpareto.cdf(
        x,
        c=xi,
        loc=0,
        scale=sigma
    )

    eps = 1e-10

    F = np.clip(F, eps, 1-eps)

    i = np.arange(1, n+1)

    A2 = (
        -n
        - np.sum(
            (2*i-1)
            *
            (
                np.log(F)
                +
                np.log(1-F[::-1])
            )
        )
        / n
    )
 

    return A2