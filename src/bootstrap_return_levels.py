import numpy as np

from scipy.stats import genpareto

from src.gpd import fit_gpd
from src.return_level import return_level


def bootstrap_return_levels(
    exceedances,
    threshold,
    rate,
    periods,
    B=2000
):
    """
    Bootstrap confidence intervals for return levels.
    """

    n = len(exceedances)

    estimates = np.zeros((B, len(periods)))

    for b in range(B):

        sample = np.random.choice(
            exceedances,
            size=n,
            replace=True
        )

        fit = fit_gpd(sample)

        sigma = fit["sigma"]
        xi = fit["xi"]

        for j, T in enumerate(periods):

            estimates[b, j] = return_level(
                sigma,
                xi,
                threshold,
                rate,
                T
            )

    lower = np.percentile(
        estimates,
        2.5,
        axis=0
    )

    upper = np.percentile(
        estimates,
        97.5,
        axis=0
    )

    mean = np.mean(
        estimates,
        axis=0
    )

    return mean, lower, upper