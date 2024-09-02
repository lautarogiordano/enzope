import warnings
import os
import pickle
import concurrent.futures

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
    NumbaPerformanceWarning,
)

from ..kernels import k_ys
from ..metrics import measures
from ..trades.ys import yard_sale
from ..utils.misc import print_progress_bar

# Filtro algunos warnings que tira numba
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


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
        seed (int, optional): Random seed. Defaults to None.

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
        deciles (list): List of deciles of the wealth distribution at each step.
        richest (list): List of the wealth of the richest agent at each step.

    Methods:
        get_opponents(): Get the opponents for each agent.
        choose_winner(i, j): Choose a winner between two agents based on their wealth.
        update_metrics(): Update the model's metrics.
        finalize(): Convert some lists to arrays.
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
        f=0.0,
        w_min=3e-17,
        w_0=None,
        r_min=0,
        r_max=1,
        measure_every=np.inf,
        upd_w_every=np.inf,
        upd_graph_every=np.inf,
        plot_every=np.inf,
        seed=None,
    ):
        self.n_agents = n_agents
        self.w_min = w_min
        self.f = f
        self.G = G if G is not None else None
        self.interaction = interaction
        self.update_w = upd_w_every
        self.update_links = upd_graph_every
        self.plot = plot_every
        # Esto es para muestrear el calculo del gini
        self.measure_every = measure_every
        # Semilla random
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Initialize n agents with random risks and wealth between (r_min, r_max]
        # and normalize wealth
        assert r_min < r_max, "r_min should be less than r_max"
        self.r = np.random.uniform(r_min, r_max, self.n_agents).astype(np.float32)
        if w_0 is not None:
            self.w = w_0
        else:
            self.w = np.random.rand(self.n_agents).astype(np.float32)
            self.w /= np.sum(self.w)
        self.w_old = np.copy(self.w)

        # Initialize measures
        self.gini = [measures.gini(self.w)]
        self.palma = [measures.palma_ratio(self.w)]
        self.n_active = [measures.num_actives(self.w, self.w_min)]
        self.liquidity = []
        self.n_frozen = (
            [measures.num_frozen(self.w, self.w_min, self.G)]
            if self.G is not None
            else []
        )
        self.deciles = [measures.deciles(self.w)]
        self.richest = [np.max(self.w)]

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

    def update_metrics(self):
        """Update model metrics."""
        self.gini.append(measures.gini(self.w))
        self.palma.append(measures.palma_ratio(self.w))
        self.n_active.append(measures.num_actives(self.w, self.w_min))
        if self.G is not None:
            self.n_frozen.append(measures.num_frozen(self.w, self.w_min, self.G))
        self.liquidity.append(measures.liquidity(self.w, self.w_old))
        self.deciles.append(measures.deciles(self.w))
        self.richest.append(np.max(self.w))

    def finalize(self):
        """Convert some lists to arrays."""
        self.deciles = np.array(self.deciles)

    def run(self, steps, verbose=False):  # sourcery skip: remove-unnecessary-else
        """
        Main MC loop

        Args:
            steps (int): The number of steps to run the MC loop.

        """
        for mcs in range(1, steps + 1):
            if verbose:
                print_progress_bar(mcs, steps)

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

                    # Do the transaction only if the loser has enough wealth
                    if self.w[loser] >= dw:
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
                self.update_metrics()

        # A function that converts some lists to arrays
        self.finalize()

    def save(self, filename="default", filepath=os.getcwd()):
        """
        Save the model's state to a file.

        Args:
            filename (str): The name of the file to save the state to. Defaults to 'default'.
            filepath (str): The path to the file. Defaults to the current working directory.
        """
        try:
            if filename == "default":
                graph = "mean_field" if self.G is None else "graph"
                filename = f"model_agents={self.n_agents}_f={self.f}_mcs={len(self.gini)}_{graph}"
            with open(os.path.join(filepath, f"{filename}.pkl"), "wb") as f:
                pickle.dump(self.__dict__, f)
        except Exception as e:
            print(f"Error saving model: {e}")

    def load(self, filename, filepath=os.getcwd()):
        """
        Load the model's state from a file.

        Args:
            filename (str): The name of the file to load the state from.
            filepath (str): The path to the file. Defaults to the current working directory.
        """
        try:
            with open(os.path.join(filepath, f"{filename}.pkl"), "rb") as f:
                self.__dict__ = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")

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
        print(f"w_min: {self.w_min}")
        print(f"Seed: {self.seed}")
        print(f"Current Gini: {measures.gini(self.w):.4f}")
        print(f"Current Actives: {measures.num_actives(self.w, self.w_min):.4f}")
        print(f"Richest Agent wealth: {self.richest[-1]:.4f}")
        print("------------------")


# Probando CPUEnsemble de gpt:
class CPUEnsemble:
    """
    A class representing an ensemble of CPU models.

    Parameters:
    - n_models (int): The number of models in the ensemble.
    - model_params (dict): The parameters for each model in the ensemble.
    - seed (int, optional): The seed for random number generation. Defaults to None.

    Attributes:
    - n_models (int): The number of models in the ensemble.
    - models (list): The list of CPUModel instances in the ensemble.
    - seed (int): The seed for random number generation.

    Methods:
    - __init__(self, n_models, model_params, seed=None): Initializes the CPUEnsemble instance.
    - run(self, steps, parallel=False): Runs the models in the ensemble for the specified number of steps.
    - save_ensemble(self, filepath=os.getcwd()): Saves the ensemble models to disk.
    - load_ensemble(self, n_models, filepath=os.getcwd()): Loads the ensemble models from disk.
    - aggregate_results(self): Aggregates the results from all models in the ensemble.
    """

    def __init__(self, n_models, model_params, seed=None):
        self.n_models = n_models
        self.models = []
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            # self.seeds = np.random.randint(0, 10000, size=n_models)
        # else:
        # self.seeds = [None] * n_models

        for i in range(n_models):
            params = model_params.copy()
            # params['seed'] = self.seed[i]
            self.models.append(CPUModel(**params))

    def run(self, steps, verbose=False, parallel=False):
        if parallel:
            # Con threads (me da mas lento en los casos que me importan)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(model.run, steps) for model in self.models]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        else:
            for i, model in enumerate(self.models):
                if verbose:
                    print_progress_bar(i, self.n_models)

                model.run(steps)

    def save_ensemble(self, filepath=os.getcwd()):
        for idx, model in enumerate(self.models):
            model.save(filename=f"model_{idx}", filepath=filepath)

    def load_ensemble(self, n_models, filepath=os.getcwd()):
        self.models = []
        for idx in range(n_models):
            model = CPUModel()
            model.load(filename=f"model_{idx}", filepath=filepath)
            self.models.append(model)

    def aggregate_results(self):
        """
        Aggregate the results of all models.

        Returns:
            A dictionary containing the mean values of gini, palma, n_active, and liquidity.
        """
        all_gini = np.array([model.gini for model in self.models])
        all_palma = np.array([model.palma for model in self.models])
        all_n_active = np.array([model.n_active for model in self.models])
        all_liquidity = np.array([model.liquidity for model in self.models])

        mean_gini = np.mean(all_gini, axis=0)
        mean_palma = np.mean(all_palma, axis=0)
        mean_n_active = np.mean(all_n_active, axis=0)
        mean_liquidity = np.mean(all_liquidity, axis=0)

        return {
            "mean_gini": mean_gini,
            "mean_palma": mean_palma,
            "mean_n_active": mean_n_active,
            "mean_liquidity": mean_liquidity,
        }


class GPUModel(object):
    """
    Represents a GPU model for simulation.

    Args:
        n_agents (int): Number of agents.
        G (Graph, optional): The graph representing the network connections between agents. Default is None, which is equivalent to a mean-field model.
        f (float, optional): Some parameter. Defaults to 0.
        w_min (float, optional): Minimum value for w. Defaults to 1e-17.
        w_0 (ndarray, optional): Initial wealth distribution. Defaults to None.
        r_min (float, optional): Minimum value for the risk of the agents. Defaults to 0.
        r_max (float, optional): Maximum value for the risk of the agents. Defaults to 1.
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
        f=0.0,
        w_min=3e-17,
        w_0=None,
        r_min=0,
        r_max=1,
        tpb=32,
        bpg=512,
        stream=None,
        seed=None,
    ):
        self.n_agents = n_agents
        self.w_min = w_min
        self.G = G

        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        # Initialize n agents with random risks and wealth between (0, 1]
        # and normalize wealth
        self.w = np.random.rand(self.n_agents).astype(np.float32)
        self.r = np.random.uniform(r_min, r_max, self.n_agents).astype(np.float32)

        # Mutex for double lock
        self.m = np.zeros((self.n_agents), dtype=np.int32)
        self.f = f

        self.w = w_0 if w_0 is not None else self.w / (np.sum(self.w))
        # If we have a graph, we need more parameters for the kernel
        if self.G is not None:
            (self.c_neighs, self.neighs) = self.G.get_neighbors_array_gpu()

        self.stream = stream if stream is not None else cuda.default_stream()
        self.tpb = tpb
        self.bpg = bpg

        random_seed = (
            seed if seed is not None else np.random.randint(0, 0x7FFFFFFFFFFFFFFF)
        )
        self.rng_state = create_xoroshiro128p_states(self.n_agents, seed=random_seed)

    def run(self, steps, tpb=None, bpg=None, rng_state=None):
        """
        Perform Monte Carlo simulation.

        Args:
            steps (int): Number of simulation steps.
            tpb (int, optional): Threads per block. Defaults to None.
            bpg (int, optional): Blocks per grid. Defaults to None.
            rng_state (ndarray, optional): Random number generator state. Defaults to None.

        """
        # Set default values if not provided
        tpb = self.tpb if tpb is None else tpb
        bpg = self.bpg if bpg is None else bpg
        rng_state = self.rng_state if rng_state is None else rng_state

        with cuda.pinned(self.w):
            w_d = cuda.to_device(self.w, stream=self.stream)
            r_d = cuda.to_device(self.r, stream=self.stream)
            m_d = cuda.to_device(self.m, stream=self.stream)

            # No graph -> Mean field kernel
            if self.G is None:
                k_ys.k_ys_mcs[bpg, tpb, self.stream](  # type: ignore
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

                k_ys.k_ys_mcs_graph[bpg, tpb, self.stream](  # type: ignore
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
        get_mean_gini(): Computes the mean and standard deviation of the Gini coefficients for the models.
        get_mean_n_active(): Computes the mean and standard deviation of the number of active agents for the models.
        get_mean_n_frozen(): Computes the mean and standard deviation of the number of frozen agents for the models.
    """

    def __init__(
        self,
        n_models=1,
        n_agents=1000,
        tpb=32,
        bpg=512,
        graphs=None,
        random_seeds=None,
        **kwargs,
    ):
        self.n_streams = n_models
        self.n_agents = n_agents
        self.tpb = tpb
        self.bpg = bpg
        self.graphs = graphs
        random_seeds = (
            random_seeds
            if random_seeds is not None
            else np.random.randint(0, 0x7FFFFFFFFFFFFFFF, size=self.n_streams)
        )

        # Creation of GPU arrays
        if self.n_streams == 1:
            self.streams = [cuda.default_stream()]
        else:
            self.streams = [cuda.stream() for _ in range(self.n_streams)]

        if self.graphs is None:
            self.models = [
                GPUModel(n_agents=n_agents, stream=stream, tpb=tpb, bpg=bpg, **kwargs)
                for stream in self.streams
            ]
        else:
            self.models = [
                GPUModel(
                    n_agents=n_agents,
                    stream=stream,
                    G=self.graphs[i],
                    tpb=tpb,
                    bpg=bpg,
                    **kwargs,
                )
                for i, stream in enumerate(self.streams)
            ]

        self.rng_states = [
            create_xoroshiro128p_states(n_agents, seed=random_seeds[i])
            for i in range(self.n_streams)
        ]

    def run(self, steps, verbose=False):
        """
        Performs a Monte Carlo simulation for the specified number of steps.

        Args:
            steps (int): Number of simulation steps to perform.
        """
        for model, rng_state in zip(self.models, self.rng_states):
            model.run(steps, self.tpb, self.bpg, rng_state)

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

    def get_mean_gini(self):
        """
        Computes the mean and standard deviation of the Gini coefficients for the models.

        Returns:
            tuple: A tuple containing the mean and standard deviation of the Gini coefficients.
        """
        ginis = [measures.gini(model.w) for model in self.models]
        return np.mean(ginis), np.std(ginis)

    def get_mean_n_active(self):
        """
        Computes the mean and standard deviation of the number of active agents for the models.

        Returns:
            tuple: A tuple containing the mean and standard deviation of the number of active agents.
        """
        n_active = [measures.num_actives(model.w, model.w_min) for model in self.models]
        return np.mean(n_active), np.std(n_active)

    def get_mean_n_frozen(self):
        """
        Computes the mean and standard deviation of the number of frozen agents for the models (only works if a graph is present).

        Returns:
            tuple: A tuple containing the mean and standard deviation of the number of frozen agents.
        """

        n_frozen = [
            measures.num_frozen(model.w, model.w_min, model.G) for model in self.models
        ]
        return np.mean(n_frozen), np.std(n_frozen)
