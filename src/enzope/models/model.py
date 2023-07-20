# from ..kernels.k_ys import *
import time
import warnings

import numpy as np
from numba import cuda
from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
    NumbaPerformanceWarning,
)

from ..graphs.graph_class import GTG
from ..kernels import k_ys
from ..metrics import gpu_measures, measures
from ..trades.ys import yard_sale

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
        measure_every=np.inf,
        upd_w_every=np.inf,
        upd_graph_every=np.inf,
        plot_every=np.inf,
    ):
        self.n_agents = n_agents
        self.w_min = w_min
        self.G = G
        self.update_w = upd_w_every
        self.update_links = upd_graph_every
        self.plot = plot_every
        ## Esto es para muestrear el calculo del gini
        self.measure_every = measure_every


class CPUModel(BaseModel):
    def __init__(
        self, n_agents=100, G=None, w_0=None, f=0, **kwargs
    ):
        super().__init__(n_agents, **kwargs)
        # Initialize n agents with random risks and wealth between (0, 1]
        # and normalize wealth
        self.r = np.random.rand(self.n_agents).astype(np.float32)
        if w_0 is not None:
            self.w = w_0
        else:
            self.w = np.random.rand(self.n_agents).astype(np.float32)
            self.w /= np.sum(self.w)
        self.w_old = np.copy(self.w)
        self.f = f
        self.G = G if G is not None else None
        self.gini = [self.get_gini()]
        self.n_active = [self.get_n_actives()]
        self.liquidity = []
        self.n_frozen = []

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

    def choose_winner(self, i, j):
        p = 0.5 + self.f * ((self.w[j] - self.w[i]) / (self.w[i] + self.w[j]))
        return np.random.choice([i, j], p=[p, 1 - p])

    def get_gini(self):
        return measures.gini(self.w)

    def get_n_actives(self):
        return measures.num_actives(self.w, self.w_min)

    def get_n_frozen(self):
        return measures.num_frozen(self.w, self.w_min, self.G)

    def get_liquidity(self):
        return measures.liquidity(self.w, self.w_old)

    def MCS(self, steps):
        """
        Main MC loop
        """
        for mcs in range(1, steps):
            self.w_old[:] = self.w

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

            # After self.measure_every MCS append new Gini index
            if (mcs + 1) % self.measure_every == 0:
                self.gini.append(self.get_gini())
                self.n_active.append(self.get_n_actives())
                if self.G is not None:
                    self.n_frozen.append(self.get_n_frozen())
                self.liquidity.append(self.get_liquidity())


class GPUModel(BaseModel):
    def __init__(
        self, n_agents=100, w_0=None, f=0, tpb=32, bpg=512, stream=None, **kwargs
    ):
        super().__init__(n_agents, **kwargs)
        # Initialize n agents with random risks and wealth between (0, 1]
        # and normalize wealth
        self.w = np.random.rand(self.n_agents).astype(np.float32)
        self.r = np.random.rand(self.n_agents).astype(np.float32)
        # Mutex for double lock
        self.m = np.zeros((self.n_agents), dtype=np.int32)
        self.f = f

        self.w = w_0 if w_0 is not None else self.w / (np.sum(self.w))
        # If we have a graph, we need more parameters for the kernel
        if self.G is not None:
            (self.c_neighs, self.neighs) = self.G.get_neighbors_array_gpu()

        self.stream = stream if stream != None else cuda.default_stream()
        self.tpb = tpb
        self.bpg = bpg

    def MCS(self, steps, tpb=None, bpg=None, rng_state=None):
        with cuda.pinned(self.w):
            w_d = cuda.to_device(self.w, stream=self.stream)
            r_d = cuda.to_device(self.r, stream=self.stream)
            m_d = cuda.to_device(self.m, stream=self.stream)

            if tpb is None or bpg is None:
                tpb = self.tpb
                bpg = self.bpg

            if rng_state is None:
                random_seed = np.random.randint(0, 0x7FFFFFFFFFFFFFFF)
                rng_state = create_xoroshiro128p_states(self.n_agents, seed=random_seed)

            # No graph -> Mean field kernel
            if self.G is None:
                k_ys.k_ys_mcs[bpg, tpb, self.stream](
                    self.n_agents,
                    w_d,
                    r_d,
                    m_d,
                    self.w_min,
                    self.f,
                    steps,
                    rng_state,
                )

            # If we have a graph we run another kernel
            if self.G is not None:
                c_neighs_d = cuda.to_device(self.c_neighs, stream=self.stream)
                neighs_d = cuda.to_device(self.neighs, stream=self.stream)

                k_ys.k_ys_mcs_graph[bpg, tpb, self.stream](
                    self.n_agents,
                    w_d,
                    r_d,
                    m_d,
                    c_neighs_d,
                    neighs_d,
                    self.w_min,
                    self.f,
                    steps,
                    rng_state,
                )

                del c_neighs_d, neighs_d

            w_d.copy_to_host(self.w, self.stream)
        del w_d, r_d, m_d

        cuda.synchronize()


class GPUEnsemble:
    def __init__(
        self, n_models=1, n_agents=1000, tpb=32, bpg=512, graphs=None, **kwargs
    ):
        self.n_streams = n_models
        self.n_agents = n_agents
        self.tpb = tpb
        self.bpg = bpg
        self.graphs = graphs
        random_seeds = np.random.randint(0, 0x7FFFFFFFFFFFFFFF, size=self.n_streams)

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
            create_xoroshiro128p_states(n_agents, seed=random_seeds[i])
            for i in range(self.n_streams)
        ]

    def MCS(self, steps):
        for model, rng_state in zip(self.models, self.rng_states):
            model.MCS(steps, self.tpb, self.bpg, rng_state)

    def save_wealths(self, filepath=None):
        if filepath is None:
            raise ValueError("Insert a valid filepath with the structure 'path/name'")
        else:
            np.save(
                filepath, np.array([[self.models[i].w] for i in range(self.n_streams)])
            )


## LA DEJO COMENTADA PORQUE NO ME GUSTA. ESTO VA A SER LENTISIMO YA QUE
## CADA VEZ QUE CORTO EL KERNEL PARA CALCULAR LOS GINIS BORRO Y COPIO MEMORIA
## A LO LOCO (SIN CONTAR QUE LLAMAR A UN KERNEL VARIAS VECES YA ES MAS LENTO DE
## POR SI). PODRIA SOLUCIONARLO MAS FACIL SI ENCONTRARA ALGUNA FORMA DE SORTEAR
## EL ARRAY W DESDE DENTRO DEL DEVICE.
# class GPUEnsembleModified(GPUEnsemble):
#     """
#     Modified version of GPUEnsemble class that permits the computation of
#     some measures like the gini index, at the expense of a higher runtime
#     (10-100x slower).
#     """

#     def __init__(
#         self,
#         gini_every=np.inf,
#         n_models=1,
#         n_agents=1000,
#         tpb=32,
#         bpg=512,
#         graphs=None,
#         **kwargs
#     ):
#         super().__init__(n_models, n_agents, tpb, bpg, graphs, **kwargs)
#         self.gini_every = gini_every
#         self.ginis = [{} for _ in range(self.n_models)]

#     def MCS(self, steps):
#         run_steps = self.gini_every
#         for i in range(steps // run_steps + 1):
#             for model, rng_state in zip(self.models, self.rng_states):
#                 # We run the model run_steps times and append the ginis
#                 model.MCS(run_steps, self.tpb, self.bpg, rng_state)
#                 # self.ginis[i][t] means the gini of model i at time t
#                 self.ginis[model][i * run_steps] = measures.cupy_gini(
#                     cp.asarray(self.models[model].w)
#                 )
