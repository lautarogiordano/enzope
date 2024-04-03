# from ..kernels.k_ys import *
import time
import warnings
import os
import pickle

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


class CPUModel(object):
    """
    Represents a CPU model for simulating agent-based economic interactions.

    Args:
        n_agents (int): The number of agents in the model.
        G (Graph, optional): The graph representing the network connections between agents. Default is None, which is equivalent to a mean-field model.
        interaction (function, optional): Function that represents the interaction between two agents. Defaults to yard_sale.
        f (float, optional): A parameter used in the winner selection process.
        w_min (float, optional): Minimum value for w. Defaults to 1e-17.
        w_0 (ndarray, optional): Initial wealth distribution of the agents.
        r_min (float, optional): Minimum value for the risk of the agents. Defaults to 0.
        r_max (float, optional): Maximum value for the risk of the agents. Defaults to 1.
        measure_every (float, optional): Frequency of measuring the gini coefficient, the fraction of active agents, frozen agents (if working with a graph) and liquidity. Defaults to np.inf.
        upd_w_every (float, optional): Frequency of updating the weights of the graph. Defaults to np.inf.
        upd_graph_every (float, optional): Frequency of updating the graph. Defaults to np.inf.
        plot_every (float, optional): Frequency of plotting. Defaults to np.inf.

    Attributes:
        r (ndarray): Array of risks for each agent.
        w (ndarray): Array of wealth values for each agent.
        w_old (ndarray): Copy of the previous wealth distribution.
        f (float): The value of the parameter used in the winner selection process.
        G (Graph): The graph representing the network connections between agents.
        gini (list): List of Gini index values at each step.
        palma (list): List of Palma ratio values at each step.
        n_active (list): List of the number of active agents at each step.
        liquidity (list): List of liquidity values at each step.
        n_frozen (list): List of the number of frozen agents at each step.

    Methods:
        get_opponents(): Get the opponents for each agent.
        choose_winner(i, j): Choose a winner between two agents based on their wealth.
        get_gini(): Computes the Gini index of the current wealth distribution.
        get_palma_ratio(): Computes the Palma ratio of the current wealth distribution.
        get_n_actives(): Computes the number of active agents.
        get_n_frozen(): Computes the number of frozen agents.
        get_liquidity(): Computes the liquidity value.
        MCS(steps): Run the main Monte Carlo loop.
        save(filename, filepath): Save the model's state to a Pickle file.
        load(filename, filepath): Load the model's state from a Pickle file.
        info(): Print information about the model.

    """

    def __init__(
        self,
        n_agents=100,
        G=None,
        interaction=yard_sale,
        f=0,
        w_min=3e-17,
        w_0=None,
        r_min=0,
        r_max=1,
        measure_every=np.inf,
        upd_w_every=np.inf,
        upd_graph_every=np.inf,
        plot_every=np.inf,
    ):
        self.n_agents = n_agents
        self.w_min = w_min
        # Initialize n agents with random risks and wealth between (0, 1]
        # and normalize wealth
        assert(r_min < r_max)
        self.r = np.random.uniform(r_min, r_max, self.n_agents).astype(np.float32)
        if w_0 is not None:
            self.w = w_0
        else:
            self.w = np.random.rand(self.n_agents).astype(np.float32)
            self.w /= np.sum(self.w)
        self.w_old = np.copy(self.w)
        self.f = f
        self.G = G if G is not None else None
        self.interaction = interaction
        self.update_w = upd_w_every
        self.update_links = upd_graph_every
        self.plot = plot_every
        ## Esto es para muestrear el calculo del gini
        self.measure_every = measure_every

        self.gini = [self.get_gini()]
        self.palma = [self.get_palma_ratio()]
        self.n_active = [self.get_n_actives()]
        self.liquidity = []
        self.n_frozen = [self.get_n_frozen()] if self.G is not None else []

    def get_opponents(self):
        """
        Get an array of opponents for each agent.

        If self.G is None, generate a random array of opponents where each element
        represents the index of an opponent for the corresponding agent. The generated
        array ensures that no agent is assigned itself as an opponent.

        If self.G is not None, call the `get_opponents_cpu` method of self.G to get
        the opponents array.

        Returns:
            numpy.ndarray: Array of opponents for each agent.
        """
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
        """
        Chooses a winner between two options based on their weights.

        Parameters:
            i (int): The index of the first option.
            j (int): The index of the second option.

        Returns:
            int: The index of the chosen winner.
        """
        p = 0.5 + self.f * ((self.w[j] - self.w[i]) / (self.w[i] + self.w[j]))
        winner = np.random.choice([i, j], p=[p, 1 - p])
        loser = i if winner == j else j
        return winner, loser

    def get_gini(self):
        """
        Computes the Gini index of the current wealth distribution.

        Returns:
            float: The Gini index.
        """
        return measures.gini(self.w)
    
    def get_palma_ratio(self):
        """
        Computes the Palma ratio of the current wealth distribution.

        Returns:
            float: The Palma ratio.
        """
        return measures.palma_ratio(self.w)

    def get_n_actives(self):
        """
        Computes the number of active agents.

        Returns:
            int: The number of active agents.
        """
        return measures.num_actives(self.w, self.w_min)

    def get_n_frozen(self):
        """
        Computes the number of frozen agents (only works if a graph is present).

        Returns:
            int: The number of frozen agents.
        """
        return measures.num_frozen(self.w, self.w_min, self.G)

    def get_liquidity(self):
        """
        Computes the liquidity value.

        Returns:
            float: The liquidity value.
        """
        return measures.liquidity(self.w, self.w_old)

    def MCS(self, steps):  # sourcery skip: remove-unnecessary-else
        """
        Main MC loop

        Args:
            steps (int): The number of steps to run the MC loop.

        """
        for mcs in range(1, steps):
            self.w_old[:] = self.w

            if self.G and mcs % self.plot == 0:
                self.G.plot_snapshot(self.w_min, mcs, mode="save")

            opps = self.get_opponents()

            for i, j in enumerate(opps):
                # Check both agents have w > w_min and node is not isolated
                if self.w[i] > self.w_min and self.w[j] > self.w_min and j != -1:
                    # Yard-Sale algorithm
                    dw = self.interaction(self.r[i], self.w[i], self.r[j], self.w[j])

                    winner, loser = self.choose_winner(i, j)

                    if self.w[loser] < dw:
                        continue

                    else:
                        self.w[winner] += dw
                        self.w[loser] -= dw

            # After self.update_w update weights
            if self.G and mcs % self.update_w == 0:
                self.G.update_weights(self.w)

            # Recompute the links if the network is dynamic
            if (mcs + 1) % self.update_links == 0 and self.G is not None:
                self.G.update_graph()

            # After self.measure_every MCS append new Gini index
            if (mcs + 1) % self.measure_every == 0:
                self.gini.append(self.get_gini())
                self.palma.append(self.get_palma_ratio())
                self.n_active.append(self.get_n_actives())
                if self.G is not None:
                    self.n_frozen.append(self.get_n_frozen())
                self.liquidity.append(self.get_liquidity())

    def save(self, filename='default', filepath=os.getcwd()):
        """
        Save the model's state to a file.

        Args:
            filename (str): The name of the file to save the state to. Defaults to 'default'.
            filepath (str): The path to the file. Defaults to the current working directory.
        """
        if filename == 'default':
            graph = 'mean_field' if self.G is None else 'graph'
            filename = f"model_agents={self.n_agents}_f={self.f}_mcs={len(self.gini)}_{graph}"
        with open(os.path.join(filepath, f'{filename}.pkl'), 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, filename, filepath=os.getcwd()):
        """
        Load the model's state from a file.

        Args:
            filename (str): The name of the file to load the state from.
            filepath (str): The path to the file. Defaults to the current working directory.
        """
        with open(os.path.join(filepath, f'{filename}.pkl'), 'rb') as f:
            self.__dict__ = pickle.load(f)

    def info(self):
            """
            Prints information about the model.

            This method prints various information about the model, including the number of agents,
            the graph, the interaction type, the function, the current Gini coefficient, the number
            of active agents, and the richest agent.

            Returns:
                None
            """
            print("--- Model Info ---")
            print(f"Agents: {self.n_agents}")
            print(f"Graph: {self.G}")
            print(f"Interaction: {self.interaction}")
            print(f"f: {self.f}")
            print(f"Current Gini: {self.get_gini()}")
            print(f"Current Actives: {self.get_n_actives()}")
            print(f"Richest Agent: {np.max(self.w)}")
            print("------------------")




class GPUModel(object):
    """
    Represents a GPU model for simulation.

    Args:
        n_agents (int): Number of agents.
        G (Graph, optional): The graph representing the network connections between agents. Default is None, which is equivalent to a mean-field model.
        f (float, optional): Some parameter. Defaults to 0.
        w_min (float, optional): Minimum value for w. Defaults to 1e-17.
        w_0 (ndarray, optional): Initial wealth distribution. Defaults to None.
        tpb (int, optional): Threads per block. Defaults to 32.
        bpg (int, optional): Blocks per grid. Defaults to 512.
        stream (Stream, optional): CUDA stream. Defaults to None.
        **kwargs: Additional keyword arguments.

    Attributes:
        w (ndarray): Array of agents' wealth.
        r (ndarray): Array of agents' risks.
        m (ndarray): Array of mutexes.
        f (float): Some parameter.
        stream (Stream): CUDA stream.
        tpb (int): Threads per block.
        bpg (int): Blocks per grid.

    Methods:
        MCS: Perform Monte Carlo simulation.

    """

    def __init__(
        self,
        n_agents=100,
        G=None,
        f=0,
        w_min=3e-17,
        w_0=None,
        tpb=32,
        bpg=512,
        stream=None,
    ):
        self.n_agents = n_agents
        self.w_min = w_min
        self.G = G

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
        """
        Perform Monte Carlo simulation.

        Args:
            steps (int): Number of simulation steps.
            tpb (int, optional): Threads per block. Defaults to None.
            bpg (int, optional): Blocks per grid. Defaults to None.
            rng_state (ndarray, optional): Random number generator state. Defaults to None.

        """
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
    """
    Represents an ensemble of GPU models.

    Args:
        n_models (int): Number of models in the ensemble. Default is 1.
        n_agents (int): Number of agents in each model. Default is 1000.
        tpb (int): Threads per block for GPU computation. Default is 32.
        bpg (int): Blocks per grid for GPU computation. Default is 512.
        graphs (list): List of graphs for each model. Default is None.
        **kwargs: Additional keyword arguments to be passed to the GPUModel constructor.

    Attributes:
        n_streams (int): Number of streams in the ensemble.
        n_agents (int): Number of agents in each model.
        tpb (int): Threads per block for GPU computation.
        bpg (int): Blocks per grid for GPU computation.
        graphs (list): List of graphs for each model.
        streams (list): List of CUDA streams for each model.
        models (list): List of GPUModel instances in the ensemble.
        rng_states (list): List of random number generator states for each model.

    Methods:
        MCS(steps): Performs a Monte Carlo simulation for the specified number of steps.
        save_wealths(filepath): Saves the wealths of the models to a file.
        get_gini(): Computes the mean and standard deviation of the Gini coefficients for the models.
        get_n_active(): Computes the mean and standard deviation of the number of active agents for the models.
        get_n_frozen(): Computes the mean and standard deviation of the number of frozen agents for the models.
    """

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
        """
        Performs a Monte Carlo simulation for the specified number of steps.

        Args:
            steps (int): Number of simulation steps to perform.
        """
        for model, rng_state in zip(self.models, self.rng_states):
            model.MCS(steps, self.tpb, self.bpg, rng_state)

    def save_wealths(self, filepath=None):
        """
        Saves the wealths of the models to a file.

        Args:
            filepath (str): Path to the file where the wealths will be saved.
                The file should have the structure 'path/name'.

        Raises:
            ValueError: If filepath is not provided.
        """
        if filepath is None:
            raise ValueError("Insert a valid filepath with the structure 'path/name'")
        else:
            np.save(
                filepath, np.array([[self.models[i].w] for i in range(self.n_streams)])
            )

    def get_gini(self):
        """
        Computes the mean and standard deviation of the Gini coefficients for the models.

        Returns:
            tuple: A tuple containing the mean and standard deviation of the Gini coefficients.
        """
        ginis = [measures.gini(model.w) for model in self.models]
        return np.mean(ginis), np.std(ginis)

    def get_n_active(self):
        """
        Computes the mean and standard deviation of the number of active agents for the models.

        Returns:
            tuple: A tuple containing the mean and standard deviation of the number of active agents.
        """
        n_active = [measures.num_actives(model.w, model.w_min) for model in self.models]
        return np.mean(n_active), np.std(n_active)

    def get_n_frozen(self):
        """
        Computes the mean and standard deviation of the number of frozen agents for the models (only works if a graph is present).

        Returns:
            tuple: A tuple containing the mean and standard deviation of the number of frozen agents.
        """

        n_frozen = [
            measures.num_frozen(model.w, model.w_min, model.G) for model in self.models
        ]
        return np.mean(n_frozen), np.std(n_frozen)


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
