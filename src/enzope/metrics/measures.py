import networkx as nx
import numpy as np


def gini(w):
    n = len(w)
    w_sorted = np.sort(w)
    p_cumsum = np.cumsum(w_sorted) / np.sum(w_sorted)
    B = np.sum(p_cumsum) / n
    return 1 + 1 / n - 2 * B


def liquidity(w, w_old):
    n = len(w)
    return np.sum(np.abs(w - w_old)) / (2 * n)


def num_actives(w, w_min):
    return np.mean(w > w_min)


def num_frozen(w, w_min, graph):
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


# def movility(w, w_old):
#     return np.mean(np.abs(w - w_old) > 0)