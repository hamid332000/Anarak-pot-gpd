import numpy as np
import pandas as pd
import config

def runs_declustering(df, threshold, run_length=5):
    """
    Runs declustering for POT exceedances.

    Parameters
    ----------
    df : DataFrame
        Must contain Date and Tmax columns.
    threshold : float
        Selected POT threshold.
    run_length : int
        Number of consecutive non-exceedance days required
        to terminate a cluster.

    Returns
    -------
    clusters : DataFrame
        One row per cluster.
    """

    exceed = df["Tmax"] > threshold

    clusters = []

    cluster = []

    gap = run_length

    for i in range(len(df)):

        if exceed.iloc[i]:

            if gap >= run_length and len(cluster) > 0:

                clusters.append(cluster)

                cluster = []

            cluster.append(i)

            gap = 0

        else:

            gap += 1

    if len(cluster) > 0:

        clusters.append(cluster)

    records = []

    for cid, c in enumerate(clusters, start=1):

        values = df.iloc[c]["Tmax"]

        dates = df.iloc[c]["Date"]

        imax = values.idxmax()

        records.append({

            "Cluster": cid,

            "Start": dates.iloc[0],

            "End": dates.iloc[-1],

            "Size": len(c),

            "Maximum": values.max(),
            
            "Excess": values.max() - threshold,

            "PeakDate": df.loc[imax, "Date"]

        })

    clusters = pd.DataFrame(records)

    return clusters