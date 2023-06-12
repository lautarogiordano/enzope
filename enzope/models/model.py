from ..kernels.k_ys import *
from ..trades.ys import yard_sale
from ..graphs.graph_class import GTG
import numpy as np
import cupy as cp
import time
import warnings

from numba import cuda
from numba.core.errors import (
    NumbaPerformanceWarning,
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
)

# Filtro algunos warnings que tira numba
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

from numba.cuda.random import create_xoroshiro128p_states


class BaseModel:
    def __init__(
        self,
        n_agents,
        w_min=1e-17,
        G=None,
        save_every=np.inf,
        upd_w_every=np.inf,
        upd_graph_every=np.inf,
        plot_every=np.inf,
    ):
        self.n_agents = n_agents
        self.w_min = w_min
        self.G = G
        if self.G is not None:
            self.update_w = upd_w_every
            self.update_links = upd_graph_every
            self.plot = plot_every
        ## Esto es para muestrear el calculo del gini
        self.every = save_every


class GPUModel(BaseModel):
    def __init__(self, w_0=None, f=0, stream=None, **kwargs):
        super().__init__(**kwargs)
        # Initialize n agents with random risks and wealth between (0, 1]
        # and normalize wealth
        self.w = np.random.rand(self.n_agents).astype(np.float32)
        self.r = np.random.rand(self.n_agents).astype(np.float32)
        # Mutex for double lock
        self.m = np.zeros((self.n_agents), dtype=np.int32)
        self.f = f

        if w_0 is not None:
            self.w = w_0
        else:
            self.w = self.w / (np.sum(self.w))

        # If we have a graph, we need more parameters for the kernel
        if self.G is not None:
            (
                self.n_neighs,
                self.c_neighs,
                self.neighs,
            ) = self.G.get_neighbors_array_gpu()

        self.stream = stream if stream != None else cuda.default_stream()


    def MCS(self, steps, tpb, bpg, rng_state):
        with cuda.pinned(self.w):
            w_d = cuda.to_device(self.w, stream=self.stream)
            r_d = cuda.to_device(self.r, stream=self.stream)
            m_d = cuda.to_device(self.m, stream=self.stream)

            # No graph -> Mean field kernel
            if self.G is None:
                k_ys_mcs[bpg, tpb, self.stream](
                    self.n_agents,
                    w_d,
                    r_d,
                    m_d,
                    self.w_min,
                    self.f,
                    steps,
                    rng_state,
                )

            # If we have a graph we have another kernel
            else:
                n_neighs_d = cuda.to_device(self.n_neighs, stream=self.stream)
                c_neighs_d = cuda.to_device(self.c_neighs, stream=self.stream)
                neighs_d = cuda.to_device(self.neighs, stream=self.stream)

                k_ys_mcs_graph[bpg, tpb, self.stream](
                    self.n_agents,
                    w_d,
                    r_d,
                    m_d,
                    n_neighs_d,
                    c_neighs_d,
                    neighs_d,
                    self.w_min,
                    self.f,
                    steps,
                    rng_state,
                )

                del n_neighs_d, c_neighs_d, neighs_d

            w_d.copy_to_host(self.w, self.stream)
        del w_d, r_d, m_d

        cuda.synchronize()


class GPUEnsemble:
    def __init__(
        self, n_models=1, n_agents=1000, tpb=32, bpg=256, graphs=None, **kwargs
    ):
        self.n_streams = n_models
        self.n_agents = n_agents
        self.tpb = tpb
        self.bpg = bpg
        self.graphs = graphs

        # Creation of GPU arrays
        if self.n_streams == 1:
            self.streams = [cuda.default_stream()]
        else:
            self.streams = [cuda.stream() for _ in range(self.n_streams)]

        if self.graphs is None:
            self.models = [
                GPUModel(n_agents=n_agents, stream=stream, **kwargs)
                for stream in self.streams
            ]
        else:
            self.models = [
                GPUModel(n_agents=n_agents, stream=stream, G=self.graphs[i], **kwargs)
                for i, stream in enumerate(self.streams)
            ]

        self.rng_states = [
            create_xoroshiro128p_states(n_agents, seed=time.time())
            for _ in range(self.n_streams)
        ]

    def MCS(self, steps):
        for i, (model, rng_state) in enumerate(zip(self.models, self.rng_states)):
            model.MCS(steps, self.tpb, self.bpg, rng_state)


class CPUModel(BaseModel):
    def __init__(self, G=None, w_0=None, f=0, **kwargs):
        super().__init__(**kwargs)
        # Initialize n agents with random risks and wealth between (0, 1]
        # and normalize wealth
        self.w = np.random.rand(self.n_agents).astype(np.float32)
        self.r = np.random.rand(self.n_agents).astype(np.float32)
        self.w = w_0 if w_0 is not None else self.w / (np.sum(self.w))
        self.f = f
        self.G = G if G is not None else None
        # self.gini = [self.get_gini()]
        # self.n_active = [self.get_actives()]

    def get_opponents(self):
        if self.G is None:
            random_array = np.random.randint(0, self.n_agents, self.n_agents)
            indices = np.arange(0, self.n_agents)
            # Create array of random numbers that are not equal to the index
            # If i=j then assign j'=i+1 (j'=0 if i=N-1)
            random_array = np.where(
                random_array == indices,
                (random_array + 1) % self.n_agents,
                random_array,
            )
        else:
            random_array = self.G.get_opponents_cpu()
        return random_array

    def get_gini(self):
        w = np.sort(self.w)
        p_cumsum = np.cumsum(w) / np.sum(w)
        B = np.sum(p_cumsum) / self.n_agents
        return 1 + 1 / self.n_agents - 2 * B

    def choose_winner(self, i, j):
        p = 0.5 + self.f * ((self.w[j] - self.w[i]) / (self.w[i] + self.w[j]))
        return np.random.choice([i, j], p=[p, 1 - p])

    def MCS(self, steps):
        """
        Main MC loop
        """
        for mcs in range(steps):
            if mcs % self.plot == 0:
                self.G.plot_snapshot(self.w_min, mcs, mode="save")

            opps = self.get_opponents()

            for i, j in enumerate(opps):
                # Check both agents have w > w_min and node is not isolated
                if self.w[i] > self.w_min and self.w[j] > self.w_min and j != -1:
                    # Yard-Sale algorithm
                    dw = yard_sale(self.r[i], self.w[i], self.r[j], self.w[j])

                    winner = self.choose_winner(i, j)

                    dw = np.where(winner == i, dw, -dw)

                    self.w[i] += dw
                    self.w[j] -= dw

            # After self.update_w update weights
            if mcs % self.update_w == 0:
                self.G.update_weights(self.w)

            # Recompute the links if the network is dynamic
            if (mcs + 1) % self.update_links == 0 and self.G is not None:
                self.G.update_graph()

            # After self.every MCS append new Gini index
            # if (mcs + 1) % self.every == 0:
            # self.gini.append(self.get_gini())
            # self.n_active.append(self.get_actives())
