import pandas as pd


def gpd_summary(result):
    """
    Create a summary table for the fitted GPD.
    """

    table = pd.DataFrame({

        "Parameter": [

            "Scale (σ)",
            "Shape (ξ)",
            "Log-likelihood",
            "AIC",
            "BIC",
            "Sample size"

        ],

        "Value": [

            round(result["sigma"], 4),
            round(result["xi"], 4),
            round(result["loglik"], 2),
            round(result["AIC"], 2),
            round(result["BIC"], 2),
            result["n"]

        ]

    })

    return table