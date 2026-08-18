from pathlib import Path
import pandas as pd

import config

from src.station import Station
from src.logger import setup_logger

from src.io import read_temperature_data
from src.statistics import summary_statistics

from src.plotting import plot_timeseries
from src.threshold_stability import plot_threshold_stability

from src.export import export_latex

from src.decluster import runs_declustering

from src.run_length_sensitivity import run_length_sensitivity
from src.run_length_plot import plot_run_length_sensitivity

from src.decluster_plot import plot_declustering
from src.decluster_summary import declustering_summary

from src.gpd_summary import gpd_summary
from src.gpd import fit_gpd

from src.gpd_qq import plot_gpd_qq
from src.tail_probability import plot_tail_probability

from src.return_level_summary import return_level_summary
from src.return_level_ci import return_level_ci

from src.mrl_plot import plot_mrl

from src.dependence_plot import plot_acf
from src.dependence_summary import dependence_summary

from src.threshold import (
    generate_thresholds,
    get_exceedances
)

from src.bootstrap import bootstrap_ad

from src.stationary import (
    daily_temperature_stationarity,
    annual_temperature_data,
    annual_temperature_stationarity,
    annual_pot_counts,
    pot_count_regression,
    annual_pot_magnitudes,
    pot_magnitude_stationarity,
    plot_daily_temperature_trend,
    plot_annual_temperature,
    plot_annual_pot_counts,
    plot_annual_pot_count_regression,
    plot_annual_pot_magnitudes,
)


# =====================================================
# Logger
# =====================================================

logger = setup_logger(
    config.LOGS / "analysis.log"
)

logger.info("Starting analysis")


# =====================================================
# Station information
# =====================================================

station = Station(
    config.STATION_NAME
)


print("\n==============================")
print("Station:", config.STATION_NAME)
print("Input file:", config.INPUT_FILE)
print("Threshold:", station.selected_threshold)
print("Run length:", station.run_length)
print("==============================\n")

print("Output folder:", config.OUTPUT)



# =====================================================
# Read data
# =====================================================

df = read_temperature_data(
    config.INPUT_FILE,
    config.DATE_COLUMN,
    config.VALUE_COLUMN
)


logger.info(
    "%d observations loaded",
    len(df)
)



# =====================================================
# Summary statistics
# =====================================================

summary = summary_statistics(df)


summary.to_csv(
    config.TABLES / "summary_statistics.csv"
)


export_latex(
    summary,
    config.TABLES / "summary_statistics.tex"
)


logger.info(
    "Summary table written"
)


print("AFTER SUMMARY")



# =====================================================
# THRESHOLD STAGE
# =====================================================

if config.ANALYSIS_STAGE == "threshold":


    print("Entered threshold search")


    thresholds = generate_thresholds(
        df["Tmax"].values,
        qmin=config.LOWER_QUANTILE,
        qmax=config.UPPER_QUANTILE,
        step=config.THRESHOLD_STEP,
    )


    print("Thresholds generated")



    results = []


    print(
        "******** USING THRESHOLD TEST ********"
    )


    # =================================================
    # DEVELOPMENT MODE
    # Use only 44.3 to test quickly
    #
    # Later change to:
    #
    # for u in thresholds:
    #
    # for final paper threshold stability
    # =================================================

    for u in thresholds:


        # ---------------------------------------------
        # Raw exceedances
        # ---------------------------------------------

        raw_exc = get_exceedances(
            df["Tmax"].values,
            u
        )



        # ---------------------------------------------
        # Temporary declustering
        #
        # Run length is unknown at this stage.
        # We use 5 only to obtain independent events.
        #
        # Later in FULL stage this is replaced
        # by selected run_length from JSON.
        # ---------------------------------------------

        clusters = runs_declustering(
            df,
            threshold=u,
            run_length=5
        )


        exc = clusters["Excess"].values



        if len(exc) < config.MIN_EXCEEDANCES:

            continue



        print(
            f"Threshold {u:.2f} | "
            f"raw exceedances = {len(raw_exc)} | "
            f"independent excesses = {len(exc)}"
        )



        # ---------------------------------------------
        # Fit GPD
        # ---------------------------------------------

        fit = fit_gpd(
            exc
        )


        sigma = fit["sigma"]

        xi = fit["xi"]



        print(
            f"GPD sigma={sigma:.4f}, xi={xi:.4f}"
        )



        # ---------------------------------------------
        # Bootstrap Anderson-Darling test
        #
        # refit=True:
        # Each bootstrap sample gets a new GPD fit.
        # This is the scientifically preferred version.
        # ---------------------------------------------


        print(
            "  Bootstrap starting..."
        )


        AD, p = bootstrap_ad(
            exc,
            sigma,
            xi,
            B=config.BOOTSTRAP_AD,
            refit=True,
            seed=config.RANDOM_SEED,
            verbose=False
        )


        print(
            "  Bootstrap finished."
        )


        print(
            f"  AD = {AD:.5f}"
        )

        print(
            f"  p-value = {p:.5f}"
        )



        results.append(
            [
                u,
                len(raw_exc),
                len(exc),
                sigma,
                xi,
                AD,
                p,
            ]
        )




    # =================================================
    # Save threshold results
    # =================================================


    threshold_results = pd.DataFrame(
        results,
        columns=[
            "Threshold",
            "Exceedances",
            "Independent_Exceedances",
            "Sigma",
            "Xi",
            "AD",
            "Pvalue",
        ]
    )



    threshold_results.to_csv(
        config.TABLES / 
        "threshold_candidates.csv",
        index=False
    )



    export_latex(
        threshold_results,
        config.TABLES /
        "threshold_candidates.tex"
    )



    plot_threshold_stability(
        threshold_results,
        config.FIGURES /
        "threshold_stability.png"
    )


    logger.info(
        "Threshold search completed"
    )



    print()
    print("="*60)
    print("THRESHOLD STAGE FINISHED")
    print("="*60)
    print()


    print(
        "Review:"
    )

    print(
        " Tables/threshold_candidates.csv"
    )

    print(
        " Figures/threshold_stability.png"
    )


    print()

    print(
        "Select physical threshold"
    )

    print(
        "Save it in station_config.json"
    )

    print()


    print(
        "Then set:"
    )

    print(
        'ANALYSIS_STAGE = "full"'
    )


    print()


    raise SystemExit
    
    
    # =====================================================
# FULL POT-GPD ANALYSIS
# =====================================================


# =====================================================
# STATIONARITY DIAGNOSTICS
# =====================================================

print()
print("=" * 60)
print("STATIONARITY DIAGNOSTICS")
print("=" * 60)
print()

print(
    "Running autocorrelation-adjusted daily Tmax "
    "trend analysis..."
)

daily_stationarity = daily_temperature_stationarity(
    df,
    date_column="Date",
    value_column="Tmax"
)

daily_stationarity.to_csv(
    config.TABLES / "daily_temperature_stationarity.csv",
    index=False
)

export_latex(
    daily_stationarity,
    config.TABLES / "daily_temperature_stationarity.tex"
)

plot_daily_temperature_trend(
    df,
    config.FIGURES / "daily_temperature_trend.png",
    date_column="Date",
    value_column="Tmax"
)

logger.info(
    "Daily Tmax stationarity diagnostics written"
)


print(
    "Running annual Tmax trend analysis..."
)

annual_temperature = annual_temperature_data(
    df,
    date_column="Date",
    value_column="Tmax"
)

annual_temperature_results = annual_temperature_stationarity(
    annual_temperature
)

annual_temperature.to_csv(
    config.TABLES / "annual_temperature.csv",
    index=False
)

annual_temperature_results.to_csv(
    config.TABLES / "annual_temperature_stationarity.csv",
    index=False
)

export_latex(
    annual_temperature_results,
    config.TABLES / "annual_temperature_stationarity.tex"
)

plot_annual_temperature(
    annual_temperature,
    config.FIGURES / "annual_temperature.png"
)

logger.info(
    "Annual Tmax stationarity diagnostics written"
)

print(
    "Temperature stationarity diagnostics completed."
)
print()

# =====================================================
# Dependence diagnostics BEFORE declustering
# =====================================================

print(
    "Running dependence diagnostics BEFORE declustering..."
)


exceedances = (
    df.loc[
        df["Tmax"] > station.selected_threshold,
        "Tmax"
    ]
    -
    station.selected_threshold
)


plot_acf(
    exceedances,
    nlags=10,
    outfile=
    config.FIGURES /
    "acf_before_declustering.png"
)



summary_before = dependence_summary(
    exceedances,
    lags=10
)


summary_before.to_csv(
    config.TABLES /
    "dependence_before.csv",
    index=False
)


export_latex(
    summary_before,
    config.TABLES /
    "dependence_before.tex"
)



logger.info(
    "Dependence diagnostics before declustering written"
)


print(
    "Finished dependence diagnostics BEFORE declustering."
)




# =====================================================
# Run length sensitivity
# =====================================================


print(
    "Running run-length sensitivity..."
)


years = (
    df["Date"].dt.year.max()
    -
    df["Date"].dt.year.min()
    +
    1
)



sens = run_length_sensitivity(
    df=df,
    threshold=station.selected_threshold,
    run_lengths=[3,4,5,6,7],
    years=years
)



sens.to_csv(
    config.TABLES /
    "run_length_sensitivity.csv",
    index=False
)

sens_latex = sens.rename(
    columns={
        "RunLength": "Run length (days)",
        "Clusters": "Number of clusters",
        "Sigma": r"$\sigma$",
        "Xi": r"$\xi$",
        "RL50": "50-year return level ($^\circ$C)"
    }
)

export_latex(
    sens_latex,
    config.TABLES / "run_length_sensitivity.tex"
)

export_latex(
    sens,
    config.TABLES /
    "run_length_sensitivity.tex"
)



plot_run_length_sensitivity(
    sens,
    config.FIGURES /
    "run_length_sensitivity.png"
)



logger.info(
    "Run-length sensitivity completed"
)



print(
    "Run-length sensitivity completed."
)




# =====================================================
# Stop until user selects run length
# =====================================================


if station.run_length is None:


    print()
    print("="*60)
    print("RUN LENGTH SELECTION REQUIRED")
    print("="*60)

    print()

    print(
        "Review:"
    )

    print(
        " Tables/run_length_sensitivity.csv"
    )

    print(
        " Figures/run_length_sensitivity.png"
    )

    print()

    print(
        "Select physical run length"
    )

    print(
        "Save it in station_config.json"
    )

    print()

    logger.info(
        "Run length not selected. Analysis stopped."
    )


    raise SystemExit




# =====================================================
# Declustering
# =====================================================


print(
    "Using run length:",
    station.run_length
)



clusters = runs_declustering(
    df,
    threshold=station.selected_threshold,
    run_length=station.run_length
)



clusters.to_csv(
    config.TABLES /
    "declustered_events.csv",
    index=False
)



logger.info(
    "Declustered events table written"
)



print(
    f"{len(clusters)} independent clusters identified"
)




# =====================================================
# Declustering summary
# =====================================================


n_exceedances = (
    df["Tmax"]
    >
    station.selected_threshold
).sum()



summary = declustering_summary(
    clusters,
    n_exceedances,
    station.selected_threshold
)



summary.to_csv(
    config.TABLES /
    "declustering_summary.csv",
    index=False
)



export_latex(
    summary,
    config.TABLES /
    "declustering_summary.tex"
)



logger.info(
    "Declustering summary written"
)




# =====================================================
# Declustering plot
# =====================================================


plot_declustering(
    df,
    clusters,
    threshold=
    station.selected_threshold,
    outfile=
    config.FIGURES /
    "declustering.png"
)



logger.info(
    "Declustering figure written"
)




# =====================================================
# Dependence diagnostics AFTER declustering
# =====================================================


print(
    "Running dependence diagnostics AFTER declustering..."
)



plot_acf(
    clusters["Excess"],
    nlags=10,
    outfile=
    config.FIGURES /
    "acf_after_declustering.png"
)



summary_after = dependence_summary(
    clusters["Excess"],
    lags=10
)



summary_after.to_csv(
    config.TABLES /
    "dependence_after.csv",
    index=False
)



export_latex(
    summary_after,
    config.TABLES /
    "dependence_after.tex"
)



logger.info(
    "Dependence diagnostics after declustering written"
)





# =====================================================
# ANNUAL INDEPENDENT POT-EVENT COUNTS
# =====================================================

print(
    "Running annual independent POT-event count analysis..."
)

annual_counts = annual_pot_counts(
    clusters,
    date_column="PeakDate"
)

# First and last calendar years may be incomplete.
if len(annual_counts) > 2:
    annual_counts = annual_counts.iloc[1:-1].copy()
else:
    raise ValueError(
        "Not enough complete years for annual POT-count analysis."
    )

annual_counts.to_csv(
    config.TABLES / "annual_pot_counts.csv",
    index=False
)

(
    count_regression,
    poisson_model,
    negative_binomial_model,
    count_predictions
) = pot_count_regression(
    annual_counts
)
# This file contains the Poisson result and, when overdispersion
# is detected by the stationarity module, the Negative Binomial
# result as a separate row.
count_regression.to_csv(
    config.TABLES / "pot_count_regression.csv",
    index=False
)
plot_annual_pot_count_regression(
    count_predictions,
    config.FIGURES /
    "annual_pot_count_regression.png"
)
# Also save the individual model results separately when present.
count_regression[
    count_regression["Model"].str.lower() == "poisson"
].to_csv(
    config.TABLES / "pot_count_poisson.csv",
    index=False
)

nb_results = count_regression[
    count_regression["Model"].str.lower().str.contains("negative binomial")
]

if not nb_results.empty:
    nb_results.to_csv(
        config.TABLES / "pot_count_negative_binomial.csv",
        index=False
    )

 

export_latex(
    count_regression,
    config.TABLES / "pot_count_regression.tex"
)

plot_annual_pot_counts(
    annual_counts,
    config.FIGURES / "annual_pot_counts.png"
)

logger.info(
    "Annual independent POT-event count stationarity diagnostics written"
)


# =====================================================
# ANNUAL INDEPENDENT POT-EVENT MAGNITUDES
# =====================================================

print(
    "Running annual POT-event magnitude analysis..."
)

annual_magnitudes = annual_pot_magnitudes(
    clusters,
    date_column="PeakDate",
    excess_column="Excess"
)

if len(annual_magnitudes) > 2:
    annual_magnitudes = annual_magnitudes.iloc[1:-1].copy()
else:
    raise ValueError(
        "Not enough complete years for annual POT magnitude analysis."
    )

annual_magnitudes.to_csv(
    config.TABLES / "annual_pot_magnitudes.csv",
    index=False
)

magnitude_results = pot_magnitude_stationarity(
    annual_magnitudes
)

magnitude_results.to_csv(
    config.TABLES / "pot_magnitude_trend.csv",
    index=False
)

export_latex(
    magnitude_results,
    config.TABLES / "pot_magnitude_trend.tex"
)

plot_annual_pot_magnitudes(
    annual_magnitudes,
    config.FIGURES / "annual_pot_magnitudes.png"
)

logger.info(
    "Annual POT-event magnitude stationarity diagnostics written"
)


# =====================================================
# STATIONARITY SUMMARY
# =====================================================

print()
print("STATIONARITY SUMMARY")
print("--------------------")
print()
print("Daily Tmax:")
print(daily_stationarity.to_string(index=False))
print()
print("Annual Tmax:")
print(annual_temperature_results.to_string(index=False))
print()
print("Annual POT-event count regression:")
print(count_regression.to_string(index=False))
print()
print("Annual POT-event magnitude:")
print(magnitude_results.to_string(index=False))
print()
print("Stationarity diagnostics completed.")
print()

# =====================================================
# Fit GPD to independent excesses
# =====================================================


print(
    "Fitting GPD..."
)



excess = clusters["Excess"].values



gpd = fit_gpd(
    excess
)



gpd_table = gpd_summary(
    gpd
)



gpd_table.to_csv(
    config.TABLES /
    "gpd_summary.csv",
    index=False
)



export_latex(
    gpd_table,
    config.TABLES /
    "gpd_summary.tex"
)



logger.info(
    "GPD fitted"
)



logger.info(
    "sigma=%.4f xi=%.4f",
    gpd["sigma"],
    gpd["xi"]
)




# =====================================================
# GPD QQ plot
# =====================================================


plot_gpd_qq(
    excess,
    gpd["sigma"],
    gpd["xi"],
    config.FIGURES /
    "gpd_qq.png",
    B=config.BOOTSTRAP_QQ
)



logger.info(
    "GPD QQ plot written"
)




# =====================================================
# Tail probability plot
# =====================================================


plot_tail_probability(
    excess,
    gpd["sigma"],
    gpd["xi"],
    config.FIGURES /
    "tail_probability.png"
)



logger.info(
    "Tail probability plot written"
)





# =====================================================
# Return levels
# =====================================================


rate = (
    len(clusters)
    /
    years
)



rl = return_level_summary(
    gpd,
    station.selected_threshold,
    rate
)



rl.to_csv(
    config.TABLES /
    "return_levels.csv",
    index=False
)



export_latex(
    rl,
    config.TABLES /
    "return_levels.tex"
)



logger.info(
    "Return levels written"
)




# =====================================================
# Bootstrap return level confidence intervals
# =====================================================


print(
    "Computing return level confidence intervals..."
)



rl_ci = return_level_ci(
    excess,
    station.selected_threshold,
    rate,
    B=config.BOOTSTRAP_RETURN
)



rl_ci.to_csv(
    config.TABLES /
    "return_levels_ci.csv",
    index=False
)



export_latex(
    rl_ci,
    config.TABLES /
    "return_levels_ci.tex"
)



logger.info(
    "Return level confidence intervals written"
)





# =====================================================
# Time series plot
# =====================================================


plot_timeseries(
    df,
    config.FIGURES /
    "Tmax_timeseries.png"
)



logger.info(
    "Time series plot written"
)




# =====================================================
# Mean Residual Life plot
# =====================================================


# =====================================================
# Mean Residual Life plot
# =====================================================

mrl_thresholds = generate_thresholds(
    df["Tmax"].values,
    qmin=config.LOWER_QUANTILE,
    qmax=config.UPPER_QUANTILE,
    step=config.THRESHOLD_STEP,
)

plot_mrl(
    data=df["Tmax"].values,
    thresholds=mrl_thresholds,
    selected=station.selected_threshold,
    outfile=config.FIGURES / "mrl_plot.png",
    B=config.BOOTSTRAP_MRL
)

logger.info("MRL plot written")


# =====================================================
# Finished
# =====================================================


logger.info(
    "Milestone 1 completed"
)


print()
print("="*60)
print("FULL POT-GPD ANALYSIS FINISHED")
print("="*60)