import pandas as pd

from src.return_level import return_level


def return_level_summary(
    gpd,
    threshold,
    rate
):

    periods = [10, 20, 50]

    values = []

    for T in periods:

        z = return_level(
            gpd["sigma"],
            gpd["xi"],
            threshold,
            rate,
            T
        )

        values.append([T, round(z, 2)])

    return pd.DataFrame(
        values,
        columns=[
            "Return period (years)",
            "Return level (°C)"
        ]
    )