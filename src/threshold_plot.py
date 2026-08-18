import matplotlib.pyplot as plt


def plot_threshold_results(df, filename):

    fig, ax = plt.subplots(figsize=(7,4))

    ax.plot(
        df["Threshold"],
        df["Exceedances"],
        "-o",
        lw=1.2,
        ms=4
    )

    ax.set_xlabel("Threshold ($^\circ$C)")
    ax.set_ylabel("Number of exceedances")

    fig.tight_layout()

    fig.savefig(filename, dpi=600)

    plt.close()