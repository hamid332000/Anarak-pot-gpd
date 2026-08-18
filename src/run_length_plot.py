import matplotlib.pyplot as plt


def plot_run_length_sensitivity(df, outfile):
    """
    Plot sensitivity of the POT analysis to the declustering run length.
    """

    fig, ax = plt.subplots(
        2,
        2,
        figsize=(10, 8),
        constrained_layout=True
    )

    # ---------------------------------------
    # (a) Number of clusters
    # ---------------------------------------

    ax[0,0].plot(
        df["RunLength"],
        df["Clusters"],
        "o-",
        lw=2,
        ms=7
    )

    ax[0,0].axvline(
        5,
        color="red",
        ls="--",
        lw=1.5
    )

    ax[0,0].set_title("(a)")
    ax[0,0].set_ylabel("Independent clusters")

    # ---------------------------------------
    # (b) Scale parameter
    # ---------------------------------------

    ax[0,1].plot(
        df["RunLength"],
        df["Sigma"],
        "o-",
        lw=2,
        ms=7
    )

    ax[0,1].axvline(
        5,
        color="red",
        ls="--",
        lw=1.5
    )

    ax[0,1].set_title("(b)")
    ax[0,1].set_ylabel(r"$\sigma$")

    # ---------------------------------------
    # (c) Shape parameter
    # ---------------------------------------

    ax[1,0].plot(
        df["RunLength"],
        df["Xi"],
        "o-",
        lw=2,
        ms=7
    )

    ax[1,0].axvline(
        5,
        color="red",
        ls="--",
        lw=1.5
    )

    ax[1,0].set_title("(c)")
    ax[1,0].set_ylabel(r"$\xi$")
    ax[1,0].set_xlabel("Run length (days)")

    # ---------------------------------------
    # (d) 50-year return level
    # ---------------------------------------

    ax[1,1].plot(
        df["RunLength"],
        df["RL50"],
        "o-",
        lw=2,
        ms=7
    )

    ax[1,1].axvline(
        5,
        color="red",
        ls="--",
        lw=1.5
    )

    ax[1,1].set_title("(d)")
    ax[1,1].set_ylabel("50-year return level (°C)")
    ax[1,1].set_xlabel("Run length (days)")

    # ---------------------------------------

    for a in ax.ravel():

        a.grid(alpha=0.30)

        a.set_xticks(df["RunLength"])

    fig.suptitle(
        "Sensitivity to declustering run length",
        fontsize=16
    )

    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()