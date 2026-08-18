import pandas as pd

from src.bootstrap_return_levels import (
    bootstrap_return_levels
)


def return_level_ci(
    exceedances,
    threshold,
    rate,
    B=2000
):

    periods = [10,20,50,100]

    mean, lower, upper = bootstrap_return_levels(
        exceedances,
        threshold,
        rate,
        periods,
        B=B
    )

    return pd.DataFrame({

        "Return period (years)": periods,

        "Estimate (°C)": mean.round(2),

        "Lower 95% CI": lower.round(2),

        "Upper 95% CI": upper.round(2)

    })