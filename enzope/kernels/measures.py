from numba import cuda

@cuda.jit(device=True)
def compute_gini(w):
    pass

# def get_gini(self):
#         w = np.sort(self.n[:, 0])
#         p_cumsum = np.cumsum(w) / np.sum(w)
#         B = np.sum(p_cumsum) / self.N
#         return 1 + 1 / self.N - 2 * B

@cuda.jit(device=True)
def compute_n_actives(w, wmin):
    total = len(w)
    return sum(1 for wi in w if wi > wmin)/total