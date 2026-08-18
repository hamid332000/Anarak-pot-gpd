import matplotlib.pyplot as plt


def plot_declustering(df, clusters, threshold, outfile):

    fig, ax = plt.subplots(
        figsize=(12,5)
    )

    ax.plot(
        df["Date"],
        df["Tmax"],
        color="black",
        lw=1
    )

    ax.axhline(
        threshold,
        color="red",
        ls="--",
        lw=1.5,
        label="Threshold"
    )

    ax.scatter(
        clusters["PeakDate"],
        clusters["Maximum"],
        color="red",
        s=50,
        zorder=5,
        label="Cluster maxima"
    )

    ax.set_ylabel(r"$T_{\max}$ ($^\circ$C)")
    ax.set_xlabel("Year")

    ax.legend()

    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()