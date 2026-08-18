"""
Summary table for dependence diagnostics.
"""

import pandas as pd

from src.dependence import ljung_box


def dependence_summary(
    series,
    lags=10,
    alpha=0.05
):
    """
    Summarize dependence diagnostics.

    Parameters
    ----------
    series : array-like
        Declustered exceedances.

    lags : int, default=10
        Number of lags used in the Ljung--Box test.

    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    pandas.DataFrame
        Summary table.
    """

    result = ljung_box(
        series,
        lags
    )

    statistic = result["statistic"]
    pvalue = result["pvalue"]

    conclusion = (
        "No significant serial dependence"
        if pvalue > alpha
        else "Serial dependence detected"
    )

    summary = pd.DataFrame(
        {
            "Statistic": [
                "Sample size",
                "Ljung--Box lag",
                "Ljung--Box statistic",
                "p-value",
                "Significance level",
                "Conclusion",
            ],
            "Value": [
                len(series),
                lags,
                f"{statistic:.4f}",
                f"{pvalue:.4f}",
                alpha,
                conclusion,
            ],
        }
    )

    return summary