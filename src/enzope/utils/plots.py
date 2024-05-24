import numpy as np
from matplotlib import pyplot as plt
import os


def plot_distribution(w, bins=50, title=None, figsize=(8, 6), xlabel=r"$w$", **kwargs):
    """Plot the distribution of a vector of weights."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.hist(w, bins=bins, **kwargs)
    ax.set_xlabel(xlabel)
    # ax.set_ylabel('frequency')
    if title is not None:
        ax.set_title(title)
    return fig


def plot_lorenz_curve(
    w_set,
    labels,
    dpi=150,
    savefile=None,
    plot_equality=True,
    plot_uniform=True,
    **kwargs,
):
    """
    Plots the Lorenz curve for a set of wealth distributions.

    Parameters:
    - w_set (list): List of wealth distributions to plot.
    - labels (list): List of labels for each wealth distribution.
    - dpi (int, optional): Dots per inch of the plot. Defaults to 150.
    - savefile (str, optional): Filepath to save the plot. Defaults to None.
    - plot_equality (bool, optional): Whether to plot the equality curve. Defaults to True.
    - plot_uniform (bool, optional): Whether to plot the uniform distribution curve. Defaults to True.
    - **kwargs: Additional keyword arguments to customize the plot.

    Returns:
    None
    """
    fig, ax = plt.subplots(dpi=dpi)

    ax.set_ylabel("Cumulative wealth")
    ax.set_xlabel("Fraction of agents")
    ax.set_xticks(np.arange(0, 1.05, 0.05), minor=True)
    ax.set_yticks(np.arange(0, 1.05, 0.05), minor=True)

    if plot_equality:
        # Plot gini=0 curve
        ax.plot([0, 1], [0, 1], color="k", label="Total equality")
    if plot_uniform:
        # Plot uniform distribution curve
        agent = np.random.rand(10000)
        # Normalize wealth
        agent /= np.sum(agent)
        agent = np.sort(agent)
        ax.plot(
            np.linspace(0, 1, agent.shape[0]),
            np.cumsum(agent) / np.sum(agent),
            label="Initial condition",
            color="black",
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


def plot_mean_wealth_by_risk(
    wealths,
    risks,
    r_min=0,
    r_max=1,
    num_bins=100,
    error_bars=None,
    log=None,
    title=None,
):
    """
    Plots the mean wealth as a function of risk.

    Args:
        wealths (list): List of wealth values.
        risks (list): List of risk values.
        r_min (float, optional): Minimum risk value. Defaults to 0.
        r_max (float, optional): Maximum risk value. Defaults to 1.
        num_bins (int, optional): Number of bins for risk values. Defaults to 100.
        title (str, optional): Title of the plot. Defaults to None.

    Returns:
        tuple: A tuple containing the mean risk values and the mean wealth values for each bin.
    """
    # Compute the bin edges
    bin_edges = np.linspace(r_min, r_max, num_bins + 1)

    # Group wealth values (wealths) by binned risks and calculate mean wealth for each bin
    mean_wealth_by_bin = []
    std_wealth_by_bin = []
    for i in range(num_bins):
        # Filter wealth values corresponding to the current risk bin
        bin_mask = (risks >= bin_edges[i]) & (risks < bin_edges[i + 1])

        wealths_in_bin = [w for i, w in enumerate(wealths) if bin_mask[i]]

        # Calculate mean wealth for the current risk bin
        mean_wealth = np.mean(wealths_in_bin if len(wealths_in_bin) > 10 else 0)
        std_wealth = np.std(wealths_in_bin if len(wealths_in_bin) > 10 else 0)

        mean_wealth_by_bin.append(mean_wealth)
        std_wealth_by_bin.append(std_wealth / np.sqrt(1000))

    # Calculate the mean risk value for each bin
    mean_risk_by_bin = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(num_bins)]

    # Find the index of the bin with maximum wealth
    max_wealth_index = np.argmax(mean_wealth_by_bin)

    # Retrieve the mean risk and mean wealth for the bin with maximum wealth
    max_wealth_bin_risk = mean_risk_by_bin[max_wealth_index]
    max_wealth_bin_wealth = mean_wealth_by_bin[max_wealth_index]

    # Print the mean risk and mean wealth for the bin with maximum wealth
    print(
        f"El bin con riesgo medio: {max_wealth_bin_risk:.3f} es el que tiene el maximo de riqueza media: {max_wealth_bin_wealth:.5f}"
    )

    # Plot mean wealth +- std for each risk bin and mark the bin with maximum wealth
    plt.figure(figsize=(10, 6))
    plt.plot(mean_risk_by_bin, mean_wealth_by_bin)
    if error_bars:
        plt.fill_between(
            mean_risk_by_bin,
            np.array(mean_wealth_by_bin)
            - np.array(std_wealth_by_bin) / np.sqrt(error_bars),
            np.array(mean_wealth_by_bin)
            + np.array(std_wealth_by_bin) / np.sqrt(error_bars),
            alpha=0.3,
        )
    plt.scatter(
        max_wealth_bin_risk, max_wealth_bin_wealth, color="red", label="Max wealth bin"
    )
    plt.xlim(0, r_max + 0.1)
    plt.xlabel("Riesgo medio del bin")
    plt.ylabel("Riqueza promedio")

    if log:
        plt.yscale("log")
    if not title:
        plt.title(f"Riqueza promedio del bin vs Riesgo ({num_bins} bines)")
    else:
        plt.title(title)
    plt.grid(True)
    plt.show()

    return mean_risk_by_bin, mean_wealth_by_bin
