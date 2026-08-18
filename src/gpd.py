import numpy as np
from scipy.stats import genpareto


def fit_gpd(exceedances):
    """
    Fit a Generalized Pareto Distribution by maximum likelihood.

    Parameters
    ----------
    exceedances : array-like
        Excesses above the selected threshold.

    Returns
    -------
    dict
        Estimated parameters and model statistics.
    """

    xi, loc, sigma = genpareto.fit(
        exceedances,
        floc=0
    )

    loglik = np.sum(
        genpareto.logpdf(
            exceedances,
            c=xi,
            loc=0,
            scale=sigma
        )
    )

    n = len(exceedances)

    k = 2

    AIC = 2 * k - 2 * loglik

    BIC = k * np.log(n) - 2 * loglik

    return {
        "sigma": sigma,
        "xi": xi,
        "loglik": loglik,
        "AIC": AIC,
        "BIC": BIC,
        "n": n
    }