from numba import cuda
import cupy as cp

def cupy_gini(w):
    n = len(w)
    w_sorted = cp.sort(w)
    p_cumsum = cp.cumsum(w_sorted) / cp.sum(w)
    B = cp.sum(p_cumsum) / n
    return 1 + 1 / n - 2 * B


# def get_gini(self):
#         w = np.sort(self.n[:, 0])
#         p_cumsum = np.cumsum(w) / np.sum(w)
#         B = np.sum(p_cumsum) / self.N
#         return 1 + 1 / self.N - 2 * B

@cuda.jit(device=True)
def compute_n_actives(w, wmin):
    total = len(w)
    return sum(1 for wi in w if wi > wmin)/total