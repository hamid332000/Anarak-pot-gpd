import pandas as pd

from src.decluster import runs_declustering
from src.gpd import fit_gpd
from src.return_level import return_level


def run_length_sensitivity(
    df,
    threshold,
    run_lengths,
    years
):
    """
    Sensitivity analysis for the declustering run length.
    """

    results = []

    for r in run_lengths:

        clusters = runs_declustering(
            df,
            threshold=threshold,
            run_length=r
        )

        excess = clusters["Excess"].values

        fit = fit_gpd(excess)

        rate = len(clusters) / years

        rl50 = return_level(
        sigma=fit["sigma"],
        xi=fit["xi"],
        threshold=threshold,
        rate=rate,
        return_period=50
    )
        

        results.append({

            "RunLength": r,

            "Clusters": len(clusters),

            "Sigma": fit["sigma"],

            "Xi": fit["xi"],

            "RL50": rl50

        })

    return pd.DataFrame(results)