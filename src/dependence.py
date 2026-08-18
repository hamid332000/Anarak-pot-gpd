"""
Dependence diagnostics for POT exceedances.

This module provides autocorrelation diagnostics and Ljung–Box
testing for declustered exceedance series.
"""

import numpy as np

from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox


def autocorrelation(series, nlags=20):
    """
    Compute the sample autocorrelation function.

    Parameters
    ----------
    series : array-like
        Input observations.

    nlags : int
        Number of lags.

    Returns
    -------
    ndarray
        Sample autocorrelation values.
    """

    return acf(
        series,
        nlags=nlags,
        fft=False
    )

def ljung_box(series, lags=10):
    """
    Ljung–Box test for serial independence.
    """

    result = acorr_ljungbox(
        series,
        lags=[lags],
        return_df=True
    )

    return {
        "lag": lags,
        "statistic": float(result.lb_stat.iloc[0]),
        "pvalue": float(result.lb_pvalue.iloc[0]),
    }


# Backward-compatible alias
ljung_box_test = ljung_box
