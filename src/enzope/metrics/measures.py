import numpy as np
import networkx as nx


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


def num_frozen(w, w_min, G):
    n = len(w)
    n_frozen = 0
    is_frozen = 1
    actives = [i for i in range(n) if w[i] > w_min]
    for i in actives:
        is_frozen = 1
        neighbors = list(nx.all_neighbors(G, i))
        for neigh in neighbors:
            if w[neigh] > w_min:
                is_frozen = 0
        n_frozen += is_frozen
    return n_frozen / n
