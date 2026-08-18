import matplotlib.pyplot as plt

def plot_timeseries(df, filename):

    plt.rcParams.update({

        "font.family":"Times New Roman",

        "font.size":11,

        "axes.linewidth":0.8

    })

    fig, ax = plt.subplots(figsize=(8,3.5))

    ax.plot(

        df["Date"],

        df["Tmax"],

        lw=0.7

    )

    ax.set_xlabel("Year")

    ax.set_ylabel(r"$T_{\max}$ ($^\circ$C)")

    fig.tight_layout()

    fig.savefig(filename,dpi=600)

    plt.close(fig)