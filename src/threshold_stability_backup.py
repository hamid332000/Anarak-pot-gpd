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

    print("\nThreshold stability plotting:")
    print(df_plot[[
    "Threshold",
    "Independent_Exceedances",
    "Pvalue"
]])
    fig, ax = plt.subplots(
        2, 2,
        figsize=(10, 8),
        constrained_layout=True
    )


    # ==================================================
    # (a) Exceedances
    # ==================================================

    ax[0,0].plot(
        df_plot["Threshold"],
        df_plot["Exceedances"],
        "o-",
        lw=2,
        ms=5
    )

    if selected is not None:
        ax[0,0].axvline(
            selected,
            color="red",
            ls="--",
            lw=1.5
        )

    ax[0,0].set_title("(a)")
    ax[0,0].set_ylabel("Exceedances")


    # ==================================================
    # (b) Sigma
    # ==================================================

    ax[0,1].plot(
        df_plot["Threshold"],
        df_plot["Sigma"],
        "o-",
        lw=2,
        ms=5
    )

    if selected is not None:
        ax[0,1].axvline(
            selected,
            color="red",
            ls="--",
            lw=1.5
        )

    ax[0,1].set_title("(b)")
    ax[0,1].set_ylabel(r"$\sigma$")


    # ==================================================
    # (c) Xi
    # ==================================================

    ax[1,0].plot(
        df_plot["Threshold"],
        df_plot["Xi"],
        "o-",
        lw=2,
        ms=5
    )

    ax[1,0].axhline(
        0,
        color="black",
        ls=":"
    )

    if selected is not None:
        ax[1,0].axvline(
            selected,
            color="red",
            ls="--",
            lw=1.5
        )

    ax[1,0].set_title("(c)")
    ax[1,0].set_ylabel(r"$\xi$")
    ax[1,0].set_xlabel(r"Threshold ($^\circ$C)")


    # ==================================================
    # (d) Bootstrap p-value
    # ==================================================

    ax[1,1].plot(
        df_plot["Threshold"],
        df_plot["Pvalue"],
        "o-",
        lw=2,
        ms=5
    )

    ax[1,1].axhline(
        0.05,
        color="black",
        ls=":"
    )

    if selected is not None:
        ax[1,1].axvline(
            selected,
            color="red",
            ls="--",
            lw=1.5
        )

    ax[1,1].set_title("(d)")
    ax[1,1].set_ylabel(r"$p$-value")
    ax[1,1].set_xlabel(r"Threshold ($^\circ$C)")


    # ==================================================
    # Formatting
    # ==================================================

    for a in ax.ravel():
        a.grid(alpha=0.30)


    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()