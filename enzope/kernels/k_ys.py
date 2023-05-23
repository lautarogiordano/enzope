from .locks import *
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32


# Yard-Sale kernel for mean-field run
@cuda.jit
def k_ys_mcs(n_agents, w, r, mutex, wmin, f, mcs, rng_state):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    if tid >= n_agents:
        return

    for t in range(mcs):
        for i in range(tid, n_agents, stride):
            opp = int(xoroshiro128p_uniform_float32(rng_state, i) * (n_agents))

            # Check if both agents have enough wealth to exchange
            if opp != i and w[i] > wmin and w[opp] > wmin:
                double_lock(mutex, i, opp)

                dw = min(r[i] * w[i], r[opp] * w[opp])

                # Compute the probability of winning
                prob_win = xoroshiro128p_uniform_float32(rng_state, i)
                p = 0.5 + f * (w[opp] - w[i]) / (w[i] + w[opp])
                # Determine the winner and loser
                dw = dw if prob_win <= p else -dw

                w[i] += dw
                w[opp] -= dw

                double_unlock(mutex, i, opp)

        cuda.syncthreads()


# Yard-Sale kernel in a complex network
@cuda.jit
def k_ys_mcs_graph(
    n_agents, w, r, mutex, n_neighs, cum_neighs, neighs, wmin, f, mcs, rng_state
):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    if tid >= n_agents:
        return

    for t in range(mcs):
        for i in range(tid, n_agents, stride):
            if n_neighs[i] != 0:
                # Choose random index between 0 and n_neighs - 1
                rand_neigh = int(
                    xoroshiro128p_uniform_float32(rng_state, i) * (n_neighs[i])
                )
                # Assigns i's opponent to his rand_neigh'th neighbor
                opp = neighs[cum_neighs[i] + rand_neigh]

                # Check if both agents have enough wealth to exchange
                if w[i] > wmin and w[opp] > wmin:
                    double_lock(mutex, i, opp)

                    dw = min(r[i] * w[i], r[opp] * w[opp])

                    # Compute the probability of winning
                    prob_win = xoroshiro128p_uniform_float32(rng_state, i)
                    p = 0.5 + f * (w[opp] - w[i]) / (w[i] + w[opp])
                    # Determine the winner and loser
                    dw = dw if prob_win <= p else -dw

                    w[i] += dw
                    w[opp] -= dw

                    double_unlock(mutex, i, opp)

        cuda.syncthreads()
