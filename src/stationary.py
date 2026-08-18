"""
Stationarity diagnostics for the POT-GPD analysis.

This module provides:

1. Daily Tmax trend analysis using an autocorrelation-adjusted
   Mann-Kendall test and Sen's slope.

2. Annual mean Tmax trend analysis.

3. Annual maximum Tmax trend analysis.

4. Annual independent POT-event count analysis using:
       - Poisson regression
       - Negative-binomial regression when overdispersion is present

5. Annual independent POT-event magnitude diagnostics using:
       - annual mean excess
       - annual maximum excess

6. Diagnostic plots for the above analyses.

Important
---------
The annual POT event-count analysis uses the FINAL declustered
events produced by the selected threshold and selected run length.
Therefore, it directly addresses whether the expected annual
rate of independent POT events changes with time.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm


# ============================================================
# Basic utilities
# ============================================================

def _clean_series(series):
    """
    Convert input to a clean numeric numpy array.
    """
    x = pd.Series(series).astype(float)
    x = x.replace([np.inf, -np.inf], np.nan).dropna()

    return x.to_numpy()


def _set_integer_year_ticks(ax, years):
    """
    Force calendar-year axes to display integer years.

    This prevents matplotlib from automatically generating
    labels such as 2012.5.
    """

    years = pd.to_numeric(
        pd.Series(years),
        errors="coerce"
    ).dropna().astype(int).unique()

    if len(years) == 0:
        return

    years = np.sort(years)

    ax.set_xticks(years)
    ax.set_xticklabels(
        [str(year) for year in years],
        rotation=45,
        ha="right"
    )


# ============================================================
# Mann-Kendall components
# ============================================================

def _mann_kendall_S(x):
    """
    Compute the Mann-Kendall S statistic.
    """

    x = _clean_series(x)

    n = len(x)

    if n < 2:
        return np.nan

    S = 0

    for i in range(n - 1):

        differences = x[i + 1:] - x[i]

        S += np.sum(differences > 0)
        S -= np.sum(differences < 0)

    return float(S)


def _mk_variance(x):
    """
    Variance of the Mann-Kendall S statistic.

    Ties are accounted for.
    """

    x = _clean_series(x)

    n = len(x)

    if n < 2:
        return np.nan

    _, counts = np.unique(
        x,
        return_counts=True
    )

    var_s = (
        n * (n - 1) * (2 * n + 5)
        -
        np.sum(
            counts
            * (counts - 1)
            * (2 * counts + 5)
        )
    ) / 18.0

    return float(var_s)


def _autocorrelation_adjusted_variance(
    x,
    max_lag=None
):
    """
    Estimate an autocorrelation-adjusted variance for the
    Mann-Kendall statistic.

    The adjustment inflates the ordinary MK variance according
    to serial autocorrelation in the ranked observations.

    This is intended as a practical autocorrelation-adjusted
    MK diagnostic for daily temperature data.
    """

    x = _clean_series(x)

    n = len(x)

    if n < 10:
        return _mk_variance(x), 0

    ranks = pd.Series(x).rank(
        method="average"
    ).to_numpy()

    if max_lag is None:
        max_lag = min(
            20,
            n // 4
        )

    max_lag = max(
        1,
        int(max_lag)
    )

    acf_values = acf(
        ranks,
        nlags=max_lag,
        fft=False
    )

    # Effective-variance correction.
    #
    # We use the positive autocorrelations because they
    # inflate the variance of the trend statistic.
    correction = 1.0

    for k in range(1, len(acf_values)):

        rho = acf_values[k]

        if rho > 0:

            correction += (
                2.0
                * (1.0 - k / n)
                * rho
            )

    correction = max(
        correction,
        1.0
    )

    var_s = _mk_variance(x)

    adjusted_var = var_s * correction

    return (
        float(adjusted_var),
        int(max_lag)
    )


def modified_mann_kendall(
    series,
    max_lag=None
):
    """
    Autocorrelation-adjusted Mann-Kendall test.

    Returns
    -------
    dict
        S statistic, adjusted variance, Z statistic,
        p-value, Kendall tau, and lag information.
    """

    x = _clean_series(series)

    n = len(x)

    if n < 2:

        return {
            "n": n,
            "S": np.nan,
            "variance": np.nan,
            "Z": np.nan,
            "pvalue": np.nan,
            "tau": np.nan,
            "adjustment_lag": np.nan,
        }

    S = _mann_kendall_S(x)

    var_s, used_lag = (
        _autocorrelation_adjusted_variance(
            x,
            max_lag=max_lag
        )
    )

    if var_s <= 0:

        Z = 0.0

    elif S > 0:

        Z = (
            S - 1
        ) / np.sqrt(var_s)

    elif S < 0:

        Z = (
            S + 1
        ) / np.sqrt(var_s)

    else:

        Z = 0.0

    pvalue = 2.0 * norm.sf(abs(Z))

    tau = (
        S
        /
        (n * (n - 1) / 2)
    )

    return {
        "n": n,
        "S": float(S),
        "variance": float(var_s),
        "Z": float(Z),
        "pvalue": float(pvalue),
        "tau": float(tau),
        "adjustment_lag": used_lag,
    }


# ============================================================
# Sen's slope
# ============================================================

def sens_slope(
    values,
    times=None
):
    """
    Compute Sen's slope.

    Parameters
    ----------
    values : array-like
        Observations.

    times : array-like, optional
        Time coordinate.

    Returns
    -------
    float
        Median pairwise slope.
    """

    y = _clean_series(values)

    if times is None:

        t = np.arange(
            len(y),
            dtype=float
        )

    else:

        t = np.asarray(times)

        if len(t) != len(y):
            raise ValueError(
                "times and values must have "
                "the same length."
            )

    slopes = []

    n = len(y)

    for i in range(n - 1):

        dt = (
            t[i + 1:]
            -
            t[i]
        )

        dy = (
            y[i + 1:]
            -
            y[i]
        )

        valid = dt != 0

        if np.any(valid):

            slopes.extend(
                (dy[valid] / dt[valid]).tolist()
            )

    if len(slopes) == 0:

        return np.nan

    return float(
        np.median(slopes)
    )


# ============================================================
# Daily Tmax stationarity
# ============================================================

def daily_temperature_stationarity(
    df,
    date_column="Date",
    value_column="Tmax"
):
    """
    Analyze trend in daily Tmax.

    The Mann-Kendall component is adjusted for serial
    autocorrelation.
    """

    data = df[
        [date_column, value_column]
    ].copy()

    data[date_column] = pd.to_datetime(
        data[date_column]
    )

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce"
    )

    data = data.dropna()

    data = data.sort_values(
        date_column
    )

    result = modified_mann_kendall(
        data[value_column].values
    )

    slope = sens_slope(
        data[value_column].values,
        data[date_column].map(
            pd.Timestamp.toordinal
        ).values
    )

    result["Sen_slope_per_day"] = slope

    result["Sen_slope_per_year"] = (
        slope * 365.25
    )

    return pd.DataFrame(
        {
            "Statistic": [
                "Number of daily observations",
                "MK S",
                "Autocorrelation-adjusted variance",
                "Z",
                "p-value",
                "Kendall tau",
                "Sen slope (°C/day)",
                "Sen slope (°C/year)",
                "Maximum lag used for adjustment",
            ],
            "Value": [
                result["n"],
                result["S"],
                result["variance"],
                result["Z"],
                result["pvalue"],
                result["tau"],
                result["Sen_slope_per_day"],
                result["Sen_slope_per_year"],
                result["adjustment_lag"],
            ],
        }
    )


# ============================================================
# Complete annual data
# ============================================================

def annual_temperature_data(
    df,
    date_column="Date",
    value_column="Tmax"
):
    """
    Create annual mean and annual maximum Tmax.

    The first and last calendar years are excluded because
    they may be incomplete.
    """

    data = df[
        [date_column, value_column]
    ].copy()

    data[date_column] = pd.to_datetime(
        data[date_column]
    )

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce"
    )

    data = data.dropna()

    data["Year"] = (
        data[date_column]
        .dt.year
    )

    years = sorted(
        data["Year"].unique()
    )

    if len(years) <= 2:

        raise ValueError(
            "At least three calendar years "
            "are required."
        )

    # Explicitly exclude first and last year.
    complete_years = years[1:-1]

    data = data[
        data["Year"].isin(
            complete_years
        )
    ]

    annual = (
        data
        .groupby("Year")[value_column]
        .agg(
            Mean="mean",
            Maximum="max",
            Observations="count"
        )
        .reset_index()
    )

    return annual


# ============================================================
# Annual Tmax stationarity
# ============================================================

def annual_temperature_stationarity(
    annual
):
    """
    Trend analysis of annual mean and annual maximum Tmax.
    """

    rows = []

    for variable in [
        "Mean",
        "Maximum"
    ]:

        x = annual[variable].values

        mk = modified_mann_kendall(
            x,
            max_lag=3
        )

        slope = sens_slope(
            x,
            annual["Year"].values
        )

        rows.append(
            {
                "Variable": variable,
                "N_years": len(x),
                "MK_S": mk["S"],
                "Z": mk["Z"],
                "pvalue": mk["pvalue"],
                "Kendall_tau": mk["tau"],
                "Sen_slope_degC_per_year": slope,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Annual independent POT event counts
# ============================================================

def annual_pot_counts(
    clusters,
    date_column="PeakDate",
    start_year=None,
    end_year=None
):
    """
    Count independent POT clusters by year.

    The first and last calendar years represented in the
    original data should be removed by the caller if they
    are incomplete.

    Important
    ---------
    Years with zero independent POT events are explicitly
    retained when start_year and end_year are supplied.
    This is essential for count regression because a year
    with zero events is a real observation and must not be
    omitted.
    """

    data = clusters.copy()

    data[date_column] = pd.to_datetime(
        data[date_column]
    )

    data["Year"] = (
        data[date_column]
        .dt.year
    )

    annual = (
        data
        .groupby("Year")
        .size()
        .rename("POT_Events")
        .reset_index()
    )

    # --------------------------------------------------------
    # Preserve zero-event years.
    # --------------------------------------------------------

    if start_year is not None and end_year is not None:

        all_years = pd.DataFrame(
            {
                "Year": np.arange(
                    int(start_year),
                    int(end_year) + 1
                )
            }
        )

        annual = (
            all_years
            .merge(
                annual,
                on="Year",
                how="left"
            )
        )

        annual["POT_Events"] = (
            annual["POT_Events"]
            .fillna(0)
            .astype(int)
        )

    return annual


# ============================================================
# Poisson / Negative Binomial regression
# ============================================================

def pot_count_regression(
    annual_counts,
    overdispersion_threshold=1.5
):
    """
    Test whether the expected annual number of independent
    POT events changes with year.

    First fits Poisson regression.

    If substantial overdispersion is detected, a negative
    binomial regression is also fitted and reported.

    Model:

        log(lambda_t) = beta_0 + beta_1 * year_centered

    The coefficient beta_1 measures the temporal change in
    the expected annual event rate.

    The predictor is the calendar year, centered around the
    mean year. Centering does NOT change the trend test;
    it only makes the intercept numerically easier to interpret.

    Returns
    -------
    results : pandas.DataFrame
        Model comparison table.

    poisson_model : statsmodels result
        Fitted Poisson GLM.

    negative_binomial_model : statsmodels result or None
        Fitted NB GLM if overdispersion is substantial.

    predictions : pandas.DataFrame
        Observed counts and fitted mean counts from the
        selected model(s).
    """

    data = annual_counts.copy()

    data["Year"] = pd.to_numeric(
        data["Year"],
        errors="coerce"
    )

    data["POT_Events"] = pd.to_numeric(
        data["POT_Events"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "Year",
            "POT_Events"
        ]
    )

    if len(data) < 4:

        raise ValueError(
            "At least four annual observations are "
            "required for POT count regression."
        )

    data["POT_Events"] = (
        data["POT_Events"]
        .astype(int)
    )

    data["Year_centered"] = (
        data["Year"]
        -
        data["Year"].mean()
    )

    X = sm.add_constant(
        data["Year_centered"]
    )

    y = data["POT_Events"]

    # --------------------------------------------------------
    # Poisson regression
    # --------------------------------------------------------

    poisson = sm.GLM(
        y,
        X,
        family=sm.families.Poisson()
    ).fit()

    if poisson.df_resid > 0:

        pearson_dispersion = (
            poisson.pearson_chi2
            /
            poisson.df_resid
        )

    else:

        pearson_dispersion = np.nan

    poisson_beta = (
        poisson.params["Year_centered"]
    )

    poisson_se = (
        poisson.bse["Year_centered"]
    )

    poisson_z = (
        poisson.tvalues["Year_centered"]
    )

    poisson_p = (
        poisson.pvalues["Year_centered"]
    )

    poisson_rate_change = (
        np.exp(poisson_beta) - 1.0
    ) * 100.0

    rows = [
        {
            "Model": "Poisson",
            "Coefficient": poisson_beta,
            "Std_Error": poisson_se,
            "Test_Statistic": poisson_z,
            "pvalue": poisson_p,
            "Dispersion": pearson_dispersion,
            "AIC": poisson.aic,
            "LogLikelihood": poisson.llf,
            "Rate_change_percent_per_year":
                poisson_rate_change,
            "NB_alpha": np.nan,
        }
    ]

    # --------------------------------------------------------
    # Negative binomial if overdispersion is substantial
    # --------------------------------------------------------

    negative_binomial = None

    if (
        np.isfinite(pearson_dispersion)
        and
        pearson_dispersion > overdispersion_threshold
    ):

        negative_binomial = sm.GLM(
            y,
            X,
            family=sm.families.NegativeBinomial()
        ).fit()

        beta_nb = (
            negative_binomial
            .params["Year_centered"]
        )

        se_nb = (
            negative_binomial
            .bse["Year_centered"]
        )

        z_nb = (
            negative_binomial
            .tvalues["Year_centered"]
        )

        p_nb = (
            negative_binomial
            .pvalues["Year_centered"]
        )

        rate_change_nb = (
            np.exp(beta_nb) - 1.0
        ) * 100.0

        rows.append(
            {
                "Model": "Negative binomial",
                "Coefficient": beta_nb,
                "Std_Error": se_nb,
                "Test_Statistic": z_nb,
                "pvalue": p_nb,
                "Dispersion": pearson_dispersion,
                "AIC": negative_binomial.aic,
                "LogLikelihood": negative_binomial.llf,
                "Rate_change_percent_per_year":
                    rate_change_nb,
                "NB_alpha":
                    negative_binomial.scale,
            }
        )

    # --------------------------------------------------------
    # Fitted annual counts
    #
    # Poisson fitted values are always provided.
    # NB fitted values are provided when NB is fitted.
    # --------------------------------------------------------

    predictions = data[
        [
            "Year",
            "POT_Events"
        ]
    ].copy()

    predictions["Poisson_Fitted"] = (
        poisson.predict(X)
    )

    if negative_binomial is not None:

        predictions["NB_Fitted"] = (
            negative_binomial.predict(X)
        )

    return (
        pd.DataFrame(rows),
        poisson,
        negative_binomial,
        predictions
    )


# ============================================================
# POT magnitude diagnostics
# ============================================================

def annual_pot_magnitudes(
    clusters,
    date_column="PeakDate",
    excess_column="Excess"
):
    """
    Calculate annual mean and annual maximum independent
    POT excess.
    """

    data = clusters.copy()

    data[date_column] = pd.to_datetime(
        data[date_column]
    )

    data[excess_column] = pd.to_numeric(
        data[excess_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            date_column,
            excess_column
        ]
    )

    data["Year"] = (
        data[date_column]
        .dt.year
    )

    annual = (
        data
        .groupby("Year")[excess_column]
        .agg(
            Mean_Excess="mean",
            Maximum_Excess="max",
            Events="count"
        )
        .reset_index()
    )

    return annual


def pot_magnitude_stationarity(
    annual
):
    """
    Trend tests for annual POT excess magnitudes.
    """

    rows = []

    for variable in [
        "Mean_Excess",
        "Maximum_Excess"
    ]:

        x = annual[variable].values

        mk = modified_mann_kendall(
            x,
            max_lag=3
        )

        slope = sens_slope(
            x,
            annual["Year"].values
        )

        rows.append(
            {
                "Variable": variable,
                "N_years": len(x),
                "MK_S": mk["S"],
                "Z": mk["Z"],
                "pvalue": mk["pvalue"],
                "Kendall_tau": mk["tau"],
                "Sen_slope_degC_per_year": slope,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Plot daily trend
# ============================================================

def plot_daily_temperature_trend(
    df,
    outfile,
    date_column="Date",
    value_column="Tmax"
):
    """
    Plot daily Tmax with a linear trend line.
    """

    data = df[
        [date_column, value_column]
    ].copy()

    data[date_column] = pd.to_datetime(
        data[date_column]
    )

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce"
    )

    data = data.dropna()

    x = (
        data[date_column]
        .map(pd.Timestamp.toordinal)
        .values
    )

    y = data[value_column].values

    coef = np.polyfit(
        x,
        y,
        1
    )

    trend = (
        coef[0] * x
        +
        coef[1]
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        data[date_column],
        y,
        lw=0.6,
        alpha=0.5
    )

    ax.plot(
        data[date_column],
        trend,
        lw=2
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Tmax (°C)")
    ax.set_title(
        "Daily Tmax and Linear Trend"
    )

    ax.grid(
        alpha=0.3
    )

    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# Plot annual Tmax
# ============================================================

def plot_annual_temperature(
    annual,
    outfile
):
    """
    Plot annual mean and annual maximum Tmax.

    The annual maximum must always be greater than or equal
    to the annual mean for the same year. This plot therefore
    also provides a useful visual check on the annual summary.
    """

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        annual["Year"],
        annual["Mean"],
        "o-",
        label="Annual mean Tmax"
    )

    ax.plot(
        annual["Year"],
        annual["Maximum"],
        "o-",
        label="Annual maximum Tmax"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        "Annual Tmax Statistics"
    )

    # --------------------------------------------------------
    # Use integer calendar years only.
    # --------------------------------------------------------

    _set_integer_year_ticks(
        ax,
        annual["Year"]
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# Plot annual POT counts
# ============================================================

def plot_annual_pot_counts(
    annual_counts,
    outfile
):
    """
    Plot annual number of independent POT events.
    """

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        annual_counts["Year"],
        annual_counts["POT_Events"],
        "o-",
        lw=1.8
    )

    ax.set_xlabel("Year")
    ax.set_ylabel(
        "Independent POT events"
    )

    ax.set_title(
        "Annual Number of Independent POT Events"
    )

    _set_integer_year_ticks(
        ax,
        annual_counts["Year"]
    )

    ax.grid(
        alpha=0.3
    )

    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)

# ============================================================
# Plot annual POT counts with fitted regression models
# ============================================================

def plot_annual_pot_count_regression(
    predictions,
    outfile
):
    """
    Plot observed annual independent POT-event counts together
    with the fitted Poisson and, when available, Negative
    Binomial regression curves.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Must contain:

            Year
            POT_Events
            Poisson_Fitted

        and optionally:

            NB_Fitted

    outfile : path-like
        Output path for the figure.
    """

    data = predictions.copy()

    data["Year"] = pd.to_numeric(
        data["Year"],
        errors="coerce"
    )

    data["POT_Events"] = pd.to_numeric(
        data["POT_Events"],
        errors="coerce"
    )

    data["Poisson_Fitted"] = pd.to_numeric(
        data["Poisson_Fitted"],
        errors="coerce"
    )

    if "NB_Fitted" in data.columns:

        data["NB_Fitted"] = pd.to_numeric(
            data["NB_Fitted"],
            errors="coerce"
        )

    data = data.dropna(
        subset=[
            "Year",
            "POT_Events",
            "Poisson_Fitted"
        ]
    )

    data = data.sort_values("Year")

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    # --------------------------------------------------------
    # Observed annual POT counts
    # --------------------------------------------------------

    ax.plot(
        data["Year"],
        data["POT_Events"],
        "o-",
        lw=1.8,
        ms=5,
        label="Observed independent POT events"
    )

    # --------------------------------------------------------
    # Poisson fitted mean
    # --------------------------------------------------------

    ax.plot(
        data["Year"],
        data["Poisson_Fitted"],
        "--",
        lw=2,
        label="Poisson fitted mean"
    )

    # --------------------------------------------------------
    # Negative Binomial fitted mean
    # --------------------------------------------------------

    if "NB_Fitted" in data.columns:

        ax.plot(
            data["Year"],
            data["NB_Fitted"],
            "--",
            lw=2,
            label="Negative-binomial fitted mean"
        )

    ax.set_xlabel("Year")

    ax.set_ylabel(
        "Number of independent POT events"
    )

    ax.set_title(
        "Annual Independent POT Events and Fitted Count Models"
    )

    # Use integer years on the x-axis.
    ax.set_xticks(
        data["Year"].astype(int)
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)
# ============================================================
# Plot annual POT count regression
# ============================================================

def plot_annual_pot_count_regression(
    predictions,
    outfile
):
    """
    Plot observed annual independent POT counts together with
    fitted Poisson and, when available, Negative-Binomial means.
    """

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        predictions["Year"],
        predictions["POT_Events"],
        "o-",
        lw=1.8,
        label="Observed POT events"
    )

    ax.plot(
        predictions["Year"],
        predictions["Poisson_Fitted"],
        "--",
        lw=2,
        label="Poisson fitted mean"
    )

    if "NB_Fitted" in predictions.columns:

        ax.plot(
            predictions["Year"],
            predictions["NB_Fitted"],
            "--",
            lw=2,
            label="Negative-binomial fitted mean"
        )

    ax.set_xlabel("Year")
    ax.set_ylabel(
        "Independent POT events per year"
    )

    ax.set_title(
        "Annual Independent POT Events: "
        "Observed and Fitted Count Models"
    )

    _set_integer_year_ticks(
        ax,
        predictions["Year"]
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# Plot annual POT magnitudes
# ============================================================

def plot_annual_pot_magnitudes(
    annual,
    outfile
):
    """
    Plot annual mean and maximum independent POT excess.
    """

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        annual["Year"],
        annual["Mean_Excess"],
        "o-",
        label="Mean excess"
    )

    ax.plot(
        annual["Year"],
        annual["Maximum_Excess"],
        "o-",
        label="Maximum excess"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Excess above threshold (°C)")

    ax.set_title(
        "Annual Independent POT Excess Magnitudes"
    )

    _set_integer_year_ticks(
        ax,
        annual["Year"]
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)
