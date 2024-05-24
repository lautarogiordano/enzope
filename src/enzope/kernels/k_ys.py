import warnings

from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

from .locks import double_lock, double_unlock

from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
    NumbaPerformanceWarning,
)

# Filtro algunos warnings que tira numba
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


# Yard-Sale kernel for mean-field run
@cuda.jit
def k_ys_mcs(n_agents, w, r, mutex, wmin, f, mcs, rng_state):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    if tid >= n_agents:
        return

    for t in range(mcs):
        for i in range(tid, n_agents, stride):
            j = int(xoroshiro128p_uniform_float32(rng_state, i) * (n_agents))

            # Check if both agents have enough wealth to exchange
            if j != i and w[i] > wmin and w[j] > wmin:
                double_lock(mutex, i, j)

                dw = min(r[i] * w[i], r[j] * w[j])

                # Compute the probability of winning
                rand_num = xoroshiro128p_uniform_float32(rng_state, i)
                prob_win_i = 0.5 + f * (w[j] - w[i]) / (w[i] + w[j])
                # Determine the winner and loser
                dw = dw if rand_num <= prob_win_i else -dw

                w[i] += dw
                w[j] -= dw

                double_unlock(mutex, i, j)

        cuda.syncthreads()


# Yard-Sale kernel in a complex network
@cuda.jit
def k_ys_mcs_graph(n_agents, w, r, mutex, c_neighs, neighs, wmin, f, mcs, rng_state):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    if tid >= n_agents:
        return

    for t in range(mcs):
        for i in range(tid, n_agents, stride):
            # c_neighs has N+1 components, so i does not overflow.
            n_neighs = c_neighs[i + 1] - c_neighs[i]
            if n_neighs > 0:
                # Choose random index between 0 and n_neighs - 1
                rand_neigh = int(
                    xoroshiro128p_uniform_float32(rng_state, i) * (n_neighs)
                )
                # Assigns i's opponent to his rand_neigh'th neighbor
                j = neighs[c_neighs[i] + rand_neigh]

                # Check if both agents have enough wealth to exchange
                if w[i] > wmin and w[j] > wmin:
                    double_lock(mutex, i, j)

                    dw = min(r[i] * w[i], r[j] * w[j])

                    # Compute the probability of winning
                    rand_num = xoroshiro128p_uniform_float32(rng_state, i)
                    prob_win_i = 0.5 + f * (w[j] - w[i]) / (w[i] + w[j])
                    # Determine the winner and loser
                    dw = dw if rand_num <= prob_win_i else -dw

                    w[i] += dw
                    w[j] -= dw

                    double_unlock(mutex, i, j)

        cuda.syncthreads()