import numpy as np
from scipy.stats import genpareto

from src.gpd import fit_gpd
from src.adtest import ad_statistic


def bootstrap_ad(
    exceedances,
    sigma,
    xi,
    B=500,
    seed=12345,
    refit=True,
    verbose=False
):
    """
    Parametric bootstrap Anderson-Darling goodness-of-fit test.

    Parameters
    ----------
    exceedances : ndarray
        Excesses above threshold.

    sigma : float
        Fitted scale parameter.

    xi : float
        Fitted shape parameter.

    B : int
        Number of bootstrap replications.

    seed : int
        Random seed.

    refit : bool
        True  -> re-fit GPD for every bootstrap sample
        False -> use original fitted parameters
                 (diagnostic only)

    verbose : bool
        Print first few bootstrap iterations.
    """

    rng = np.random.default_rng(seed)

    observed = ad_statistic(
        exceedances,
        sigma,
        xi
    )

    n = len(exceedances)

    statistics = np.empty(B)

    if verbose:
        print("\nObserved AD =", observed)

    for b in range(B):

        sample = genpareto.rvs(
            c=xi,
            scale=sigma,
            size=n,
            random_state=rng
        )

        if refit:

            fit = fit_gpd(sample)

            s = fit["sigma"]
            x = fit["xi"]

        else:

            s = sigma
            x = xi

        statistics[b] = ad_statistic(
            sample,
            s,
            x
        )

        if verbose and b < 10:

            print(
                f"{b+1:3d}",
                "sigma =", round(s,4),
                "xi =", round(x,4),
                "AD =", round(statistics[b],4)
            )

    p = (np.sum(statistics >= observed) + 1) / (B + 1)

    if verbose:

        print("\nBootstrap summary")
        print("-----------------")
        print("Observed AD :", observed)
        print("Mean AD     :", statistics.mean())
        print("Median AD   :", np.median(statistics))
        print("Maximum AD  :", statistics.max())
        print("Minimum AD  :", statistics.min())
        print("Bootstrap p :", p)

    return observed, p