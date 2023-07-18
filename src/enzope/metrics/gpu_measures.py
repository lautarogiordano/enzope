from numba import cuda

# import cupy as cp


# def gini_cupy(w):
#     n = len(w)
#     w_sorted = cp.sort(w)
#     p_cumsum = cp.cumsum(w_sorted) / cp.sum(w)
#     B = cp.sum(p_cumsum) / n
#     return 1 + 1 / n - 2 * B


@cuda.jit(device=True)
def num_actives(w, wmin):
    n = len(w)
    return sum(1 for wi in w if wi > wmin) / n


@cuda.jit(device=True)
def liquidity(w, w_old):
    n = len(w)
    return sum(abs(w[i] - w_old[i]) for i in range(n)) / (2 * n)
