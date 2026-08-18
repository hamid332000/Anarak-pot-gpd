import numpy as np
import pandas as pd


def generate_thresholds(
        data,
        qmin=0.90,
        qmax=0.995,
        step=0.1):

    """
    Generate candidate thresholds.
    """

    lower = np.quantile(data, qmin)
    upper = np.quantile(data, qmax)

    thresholds = np.arange(lower, upper + step, step)

    thresholds = np.round(thresholds, 1)

    return np.unique(thresholds)


def get_exceedances(data, u):

    exc = data[data > u] - u

    return exc