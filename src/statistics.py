import pandas as pd

def summary_statistics(df):

    s = df["Tmax"]

    stats = {

        "Observations": len(s),

        "Missing": int(s.isna().sum()),

        "Mean": s.mean(),

        "Median": s.median(),

        "Std": s.std(),

        "Minimum": s.min(),

        "Maximum": s.max(),

        "95%": s.quantile(.95),

        "99%": s.quantile(.99)

    }

    return pd.DataFrame.from_dict(
        stats,
        orient="index",
        columns=["Value"]
    )