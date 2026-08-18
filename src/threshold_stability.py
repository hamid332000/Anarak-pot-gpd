import matplotlib.pyplot as plt
import config


def plot_threshold_stability(df, outfile):

    from src.station import Station

    station = Station(config.STATION_NAME)

    selected = station.selected_threshold

    # ==================================================
    # Remove thresholds with too few independent events
    # for stability assessment
    # ==================================================

    MIN_INDEPENDENT_EXCEEDANCES = 40

    df_plot = df[
        df["Independent_Exceedances"]
        >= MIN_INDEPENDENT_EXCEEDANCES
    ].copy()

    # Sort by threshold to guarantee correct plotting order
    df_plot = df_plot.sort_values(
        "Threshold"
    ).reset_index(drop=True)

    print("\nThreshold stability plotting:")
    print(
        df_plot[
            [
                "Threshold",
                "Independent_Exceedances",
                "Pvalue"
            ]
        ].to_string(index=False)
    )

    # ==================================================
    # Create four-panel figure
    # ==================================================

    fig, ax = plt.subplots(
        2,
        2,
        figsize=(10, 8),
        constrained_layout=True
    )

    # ==================================================
    # (a) Number of exceedances
    # ==================================================

    ax[0, 0].plot(
        df_plot["Threshold"],
        df_plot["Exceedances"],
        "o-",
        lw=2,
        ms=5
    )

    if selected is not None:

        ax[0, 0].axvline(
            selected,
            color="red",
            ls="--",
            lw=1.5
        )

    ax[0, 0].set_title("(a)")
    ax[0, 0].set_ylabel("Exceedances")
    ax[0, 0].set_xlabel(
        r"Threshold ($^\circ$C)"
    )

    # ==================================================
    # (b) GPD scale parameter
    # ==================================================

    ax[0, 1].plot(
        df_plot["Threshold"],
        df_plot["Sigma"],
        "o-",
        lw=2,
        ms=5
    )

    if selected is not None:

        ax[0, 1].axvline(
            selected,
            color="red",
            ls="--",
            lw=1.5
        )

    ax[0, 1].set_title("(b)")
    ax[0, 1].set_ylabel(
        r"$\sigma$"
    )
    ax[0, 1].set_xlabel(
        r"Threshold ($^\circ$C)"
    )

    # ==================================================
    # (c) GPD shape parameter
    # ==================================================

    ax[1, 0].plot(
        df_plot["Threshold"],
        df_plot["Xi"],
        "o-",
        lw=2,
        ms=5
    )

    ax[1, 0].axhline(
        0,
        color="black",
        ls=":"
    )

    if selected is not None:

        ax[1, 0].axvline(
            selected,
            color="red",
            ls="--",
            lw=1.5
        )

    ax[1, 0].set_title("(c)")
    ax[1, 0].set_ylabel(
        r"$\xi$"
    )
    ax[1, 0].set_xlabel(
        r"Threshold ($^\circ$C)"
    )

    # ==================================================
    # (d) Bootstrap p-value
    # ==================================================

    # Only thresholds satisfying the minimum number
    # of independent exceedances are shown.
    #
    # Therefore thresholds such as 45.4 °C, which have
    # only 33 independent exceedances, are not displayed.

    ax[1, 1].plot(
        df_plot["Threshold"],
        df_plot["Pvalue"],
        "o-",
        lw=2,
        ms=5
    )

    ax[1, 1].axhline(
        0.05,
        color="black",
        ls=":"
    )

    if selected is not None:

        ax[1, 1].axvline(
            selected,
            color="red",
            ls="--",
            lw=1.5
        )

    ax[1, 1].set_title("(d)")
    ax[1, 1].set_ylabel(
        r"$p$-value"
    )
    ax[1, 1].set_xlabel(
        r"Threshold ($^\circ$C)"
    )

    # ==================================================
    # Formatting
    # ==================================================

    for a in ax.ravel():

        a.grid(
            alpha=0.30
        )

    # ==================================================
    # Save figure
    # ==================================================

    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "\nThreshold stability figure saved to:"
    )

    print(
        outfile
    )