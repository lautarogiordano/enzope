import networkx as nx
import numpy as np


def gini(w):
    """
    Compute the Gini coefficient for a given array of weights.

    Parameters:
    w (array-like): Array of weights.

    Returns:
    float: The Gini coefficient.

    """
    n = len(w)
    w_sorted = np.sort(w)
    p_cumsum = np.cumsum(w_sorted) / np.sum(w_sorted)
    B = np.sum(p_cumsum) / n
    return 1 + 1 / n - 2 * B


def palma_ratio(w):
    """
    Compute the Palma ratio for a given array of weights. It is defined as
    the ratio of the richest 10% of the population's wealth divided by the
    poorest 40%'s share.

    Parameters:
    w (array-like): Array of weights.

    Returns:
    float: The Palma ratio.

    """
    w_sorted = np.sort(w)
    n = len(w)
    w_top = np.sum(w_sorted[-int(0.1 * n):])
    w_bot = np.sum(w_sorted[: int(0.4 * n)])
    return w_top / w_bot


def liquidity(w, w_old):
    """
    Compute the liquidity measure between two arrays.

    Parameters:
    w (array-like): The current array of values.
    w_old (array-like): The previous array of values.

    Returns:
    float: The liquidity measure.

    """
    n = len(w)
    return np.sum(np.abs(w - w_old)) / (2 * n)


def num_actives(w, w_min):
    """
    Compute the proportion of elements in array `w` that are greater than `w_min`.

    Parameters:
    w (numpy.ndarray): The input array.
    w_min (float): The threshold value.

    Returns:
    float: The proportion of elements in `w` that are greater than `w_min`.
    """
    return np.mean(w > w_min)


def num_frozen(w, w_min, graph):
    """
    Compute the ratio of frozen nodes in a graph. Frozen nodes are those whose with wealth greater than the minimum but cannot transfer wealth to any neighbor.

    Parameters:
    w (list): List of node weights.
    w_min (float): Minimum weight threshold.
    graph (NetworkX graph): Graph object.

    Returns:
    float: Ratio of frozen nodes.
    """
    n = len(w)
    n_frozen = 0
    actives = [i for i in range(n) if w[i] > w_min]
    for i in actives:
        is_frozen = 1
        neighbors = list(nx.all_neighbors(graph.G, i))
        for neigh in neighbors:
            if w[neigh] > w_min:
                is_frozen = 0
        n_frozen += is_frozen
    return n_frozen / n


def deciles(w):
    """
    Compute the deciles of the input array.

    Parameters:
    w (array-like): The input array.

    Returns:
    numpy.ndarray: The deciles of the input array.

    """
    n = len(w)
    w_sorted = np.sort(w)
    size = n // 10
    deciles = np.array([np.sum(w_sorted[i: i + size]) for i in range(0, n, size)]).tolist()
    return deciles


def top10(deciles):
    """
    Compute the ratio of the sum of the top 10% of the population to the total sum.

    Parameters:
    deciles (numpy.ndarray): The deciles of the input array.

    Returns:
    float: The ratio of the sum of the top 10% of the population to the total sum.

    """
    return deciles[:, -1] / np.sum(deciles[0, :])


def bottom50(deciles):
    """
    Compute the ratio of the sum of the bottom 50% of the population to the total sum.

    Parameters:
    deciles (numpy.ndarray): The deciles of the input array.

    Returns:
    float: The ratio of the sum of the bottom 50% of the population to the total sum.

    """
    return np.sum(deciles[:, :5], axis=1) / np.sum(deciles[0, :])


def middle40(deciles):
    """
    Compute the ratio of the sum of the middle 40% of the population to the total sum.

    Parameters:
    deciles (numpy.ndarray): The deciles of the input array.

    Returns:
    float: The ratio of the sum of the middle 40% of the population (above the bottom 50% and below top 10%) to the total sum.

    """
    return np.sum(deciles[:, 4:9], axis=1) / np.sum(deciles[0, :])


def r1(x):
    # Esta se usa para generar las gtgs poniengo GTG(..., p_dist=measures.r1)
    return 1 / x
