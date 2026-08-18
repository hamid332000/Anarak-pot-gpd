import numpy as np


def return_level(
    sigma,
    xi,
    threshold,
    rate,
    return_period
):
    """
    Return level for the POT-GPD model.

    Parameters
    ----------
    sigma : float
        GPD scale parameter.
    xi : float
        GPD shape parameter.
    threshold : float
        Selected threshold.
    rate : float
        Mean number of independent exceedances per year.
    return_period : float
        Return period (years).

    Returns
    -------
    float
        Return level.
    """

    m = rate * return_period

    if abs(xi) < 1e-8:

        return threshold + sigma * np.log(m)

    return threshold + sigma / xi * (m**xi - 1)