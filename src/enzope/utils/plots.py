import numpy as np
from matplotlib import pyplot as plt
import os

def distribution(w, bins=50, title=None, figsize=(8, 6), xlabel=r'$w$', **kwargs):
    """Plot the distribution of a vector of weights."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.hist(w, bins=bins, **kwargs)
    ax.set_xlabel(xlabel)
    # ax.set_ylabel('frequency')
    if title is not None:
        ax.set_title(title)
    return fig

def lorenz_curve(w_set, labels, savefile=None, **kwargs):
    fig, ax = plt.subplots(dpi=300)

    ax.set_ylabel("Cumulative wealth")
    ax.set_xlabel("Fraction of agents")
    ax.set_xticks(np.arange(0, 1.05, 0.05), minor=True)
    ax.set_yticks(np.arange(0, 1.05, 0.05), minor=True)

    # Plot gini=0 curve
    ax.plot([0, 1], [0, 1], color="k", label="Total equality")
    # Plot uniform distribution curve
    agent = np.random.rand(10000)
    # Normalize wealth
    agent /= np.sum(agent)
    agent = np.sort(agent)
    ax.plot(
        np.linspace(0, 1, agent.shape[0]),
        np.cumsum(agent) / np.sum(agent),
        label="Initial condition", color="black",
        linestyle="--",
    )

    for w, label in zip(w_set, labels):

        w = np.sort(w)

        ax.plot(
            np.linspace(0, 1, w.shape[0]),
            np.cumsum(w) / np.sum(w),
            label=label,
        )
        ax.grid(which="minor", alpha=0.5, linestyle="--")
        ax.grid(which="major", alpha=1, linestyle="-")

    ax.legend(fontsize=8)

    plt.show()
    if savefile:
        fig.savefig(os.path.join(savefile, "lorenz_curve"), format="png")