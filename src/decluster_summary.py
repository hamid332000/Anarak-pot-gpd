import pandas as pd
import config


def declustering_summary(
    clusters,
    n_exceedances,
    threshold,
):
    """
    Create summary statistics for the declustering stage.

    Parameters
    ----------
    clusters : pandas.DataFrame
        Output of runs_declustering().
    n_exceedances : int
        Number of exceedances before declustering.

    Returns
    -------
    pandas.DataFrame
        Summary table.
    """

    summary = pd.DataFrame({

        "Statistic": [

            "Selected threshold (°C)",
            "Total exceedances",
            "Independent clusters",
            "Mean cluster size",
            "Median cluster size",
            "Maximum cluster size"

        ],

        "Value": [

            threshold,
            n_exceedances,
            len(clusters),
            round(clusters["Size"].mean(), 2),
            int(clusters["Size"].median()),
            int(clusters["Size"].max())

        ]

    })

    return summary