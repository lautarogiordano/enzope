from .locks import *
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32



# Kernel de Yard-Sale
@cuda.jit
def k_ys_mcs(n_agents, w, r, mutex, cum_neighs, neighs, wmin, f, rng_states, mcs):
    tid = cuda.grid(1)

    if tid >= n_agents:
        return

    cuda.syncthreads()

    if tid == 0:
        n_neighs = cum_neighs[tid]

    else:
        n_neighs = cum_neighs[tid] - cum_neighs[tid - 1]

    for _ in range(mcs):
        if n_neighs[tid] != 0:
            # Choose random index between 0 and n_neighs - 1
            rand_neigh = int(
                xoroshiro128p_uniform_float32(rng_states, tid) * (n_neighs)
            )
            # Assigns tid's opponent to his rand_neigh'th neighbor
            opp = neighs[cum_neighs[tid] + rand_neigh]

            wi = w[tid]
            wj = w[opp]

            if wi > wmin and wj > wmin:
                double_lock(mutex, tid, opp)

                dw = min(r[tid] * wi, r[opp] * wj)

                # Compute the probability of winning
                p = 0.5 + f * abs(wi - wj) / (wi + wj)
                rand = xoroshiro128p_uniform_float32(rng_states, tid)
                # Determine the winner and loser
                dw = dw if rand <= p else -dw
                wi += dw
                wj -= dw

                double_unlock(mutex, tid, opp)

        cuda.syncthreads()
