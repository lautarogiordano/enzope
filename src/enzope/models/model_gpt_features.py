import numpy as np
import os
import pickle
import logging
from tqdm import tqdm
import measures  # Assuming measures is a module containing required metric functions

logging.basicConfig(level=logging.INFO)

class CPUModel:
    """
    Represents a CPU model for simulating agent-based economic interactions.
    """

    def __init__(
        self,
        n_agents: int = 100,
        seed: int = None,
        G=None,
        interaction=None,
        f: float = 0.0,
        w_min: float = 3e-17,
        w_0=None,
        r_min: float = 0,
        r_max: float = 1,
        measure_every: float = np.inf,
        upd_w_every: float = np.inf,
        upd_graph_every: float = np.inf,
        plot_every: float = np.inf,
    ):
        self.n_agents = n_agents
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.w_min = w_min
        self.f = f
        self.G = G if G is not None else None
        self.interaction = interaction if interaction is not None else self.default_interaction
        self.update_w = upd_w_every
        self.update_links = upd_graph_every
        self.plot = plot_every
        self.measure_every = measure_every

        # Initialize agents' risks and wealth
        assert r_min < r_max, "r_min should be less than r_max"
        self.r = np.random.uniform(r_min, r_max, self.n_agents).astype(np.float32)
        self.w = w_0.astype(np.float32) if w_0 is not None else np.random.rand(self.n_agents).astype(np.float32)
        self.w /= np.sum(self.w)
        self.w_old = np.copy(self.w)

        # Metrics
        self.gini = [measures.gini(self.w)]
        self.palma = [measures.palma_ratio(self.w)]
        self.n_active = [measures.num_actives(self.w, self.w_min)]
        self.liquidity = []
        self.n_frozen = [measures.num_frozen(self.w, self.w_min, self.G)] if self.G is not None else []

    def default_interaction(self, r_i, w_i, r_j, w_j) -> float:
        """Default interaction function (example)."""
        return min(r_i * w_i, r_j * w_j)

    def get_opponents(self) -> np.ndarray:
        """Get an array of opponents for each agent."""
        if self.G is None:
            random_array = np.random.randint(0, self.n_agents, self.n_agents)
            indices = np.arange(self.n_agents)
            random_array = np.where(random_array == indices, (random_array + 1) % self.n_agents, random_array)
        else:
            random_array = self.G.get_opponents_cpu()
        return random_array

    def choose_winner(self, i: int, j: int) -> tuple:
        """Chooses a winner between two options based on their weights."""
        p = 0.5 + self.f * ((self.w[j] - self.w[i]) / (self.w[i] + self.w[j]))
        winner = np.random.choice([i, j], p=[p, 1 - p])
        loser = j if winner == i else i
        return winner, loser

    def update_metrics(self, mcs: int):
        """Update model metrics."""
        self.gini.append(measures.gini(self.w))
        self.palma.append(measures.palma_ratio(self.w))
        self.n_active.append(measures.num_actives(self.w, self.w_min))
        if self.G is not None:
            self.n_frozen.append(measures.num_frozen(self.w, self.w_min, self.G))
        self.liquidity.append(measures.liquidity(self.w, self.w_old))

    def MCS(self, steps: int):
        """Main Monte Carlo simulation loop."""
        for mcs in tqdm(range(1, steps + 1)):
            self.w_old[:] = self.w

            if self.G and mcs % self.plot == 0:
                self.G.plot_snapshot(self.w_min, mcs, mode="save")

            opps = self.get_opponents()

            for i, j in enumerate(opps):
                if self.w[i] > self.w_min and self.w[j] > self.w_min and j != -1:
                    dw = self.interaction(self.r[i], self.w[i], self.r[j], self.w[j])
                    winner, loser = self.choose_winner(i, j)
                    if self.w[loser] >= dw:
                        self.w[winner] += dw
                        self.w[loser] -= dw

            if self.G and mcs % self.update_w == 0:
                self.G.update_weights(self.w)

            if (mcs + 1) % self.update_links == 0 and self.G is not None:
                self.G.update_graph()

            if (mcs + 1) % self.measure_every == 0:
                self.update_metrics(mcs)

    def save(self, filename: str = "default", filepath: str = os.getcwd()):
        """Save the model's state to a file."""
        try:
            if filename == "default":
                graph = "mean_field" if self.G is None else "graph"
                filename = f"model_agents={self.n_agents}_f={self.f}_mcs={len(self.gini)}_{graph}"
            with open(os.path.join(filepath, f"{filename}.pkl"), "wb") as f:
                pickle.dump(self.__dict__, f)
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load(self, filename: str, filepath: str = os.getcwd()):
        """Load the model's state from a file."""
        try:
            with open(os.path.join(filepath, f"{filename}.pkl"), "rb") as f:
                self.__dict__ = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def info(self):
        """Prints information about the model."""
        print("--- Model Info ---")
        print(f"Agents: {self.n_agents}")
        print(f"Graph: {self.G}")
        print(f"Interaction: {self.interaction}")
        print(f"f: {self.f}")
        print(f"Current Gini: {measures.gini(self.w):.4f}")
        print(f"Current Actives: {measures.num_actives(self.w, self.w_min):.4f}")
        print(f"Richest Agent wealth: {np.max(self.w)}")
        print("------------------")

import concurrent.futures
import numpy as np
from tqdm import tqdm

class CPUEnsemble:
    def __init__(self, n_models, model_params, seed=None):
        self.n_models = n_models
        self.models = []
        if seed is not None:
            np.random.seed(seed)
            seeds = np.random.randint(0, 10000, size=n_models)
        else:
            seeds = [None] * n_models
        
        for i in range(n_models):
            params = model_params.copy()
            params['seed'] = seeds[i]
            self.models.append(CPUModel(**params))

    def run(self, steps):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(model.MCS, steps) for model in self.models]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                future.result()

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
        """Aggregate results from all models in the ensemble."""
        all_gini = np.array([model.gini for model in self.models])
        all_palma = np.array([model.palma for model in self.models])
        all_n_active = np.array([model.n_active for model in self.models])
        all_liquidity = np.array([model.liquidity for model in self.models])

        mean_gini = np.mean(all_gini, axis=0)
        mean_palma = np.mean(all_palma, axis=0)
        mean_n_active = np.mean(all_n_active, axis=0)
        mean_liquidity = np.mean(all_liquidity, axis=0)

        return {
            'mean_gini': mean_gini,
            'mean_palma': mean_palma,
            'mean_n_active': mean_n_active,
            'mean_liquidity': mean_liquidity,
        }


import numpy as np
from numba import cuda
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class GPUModel:
    def __init__(self, n_agents=100, G=None, f=0, w_min=3e-17, w_0=None, r_min=0, r_max=1, tpb=32, bpg=512, stream=None, seed=None):
        self.n_agents = n_agents
        self.w_min = w_min
        self.G = G

        np.random.seed(seed)
        self.w = np.random.rand(self.n_agents).astype(np.float32)
        self.r = np.random.uniform(r_min, r_max, self.n_agents).astype(np.float32)
        self.m = np.zeros((self.n_agents), dtype=np.int32)
        self.f = f
        self.w = w_0 if w_0 is not None else self.w / np.sum(self.w)
        
        if self.G is not None:
            self.c_neighs, self.neighs = self.G.get_neighbors_array_gpu()

        self.stream = stream if stream is not None else cuda.default_stream()
        self.tpb = tpb
        self.bpg = bpg
        random_seed = seed if seed is not None else np.random.randint(0, 0x7FFFFFFFFFFFFFFF)
        self.rng_state = create_xoroshiro128p_states(self.n_agents, seed=random_seed)

    def MCS(self, steps: int, tpb: int = None, bpg: int = None, rng_state: np.ndarray = None) -> None:
        tpb = tpb or self.tpb
        bpg = bpg or self.bpg
        rng_state = rng_state or self.rng_state
        
        with cuda.pinned(self.w), cuda.pinned(self.r), cuda.pinned(self.m):
            w_d = cuda.to_device(self.w, stream=self.stream)
            r_d = cuda.to_device(self.r, stream=self.stream)
            m_d = cuda.to_device(self.m, stream=self.stream)

            if self.G is None:
                k_ys.k_ys_mcs[bpg, tpb, self.stream](self.n_agents, w_d, r_d, m_d, self.w_min, self.f, steps, rng_state)
            else:
                c_neighs_d = cuda.to_device(self.c_neighs, stream=self.stream)
                neighs_d = cuda.to_device(self.neighs, stream=self.stream)
                k_ys.k_ys_mcs_graph[bpg, tpb, self.stream](self.n_agents, w_d, r_d, m_d, c_neighs_d, neighs_d, self.w_min, self.f, steps, rng_state)
                del c_neighs_d, neighs_d

            w_d.copy_to_host(self.w, self.stream)
        del w_d, r_d, m_d
        cuda.synchronize()


class GPUEnsemble:
    def __init__(self, n_models=1, n_agents=1000, tpb=32, bpg=512, graphs=None, **kwargs):
        self.n_streams = n_models
        self.n_agents = n_agents
        self.tpb = tpb
        self.bpg = bpg
        self.graphs = graphs
        random_seeds = np.random.randint(0, 0x7FFFFFFFFFFFFFFF, size=self.n_streams)

        if self.n_streams == 1:
            self.streams = [cuda.default_stream()]
        else:
            self.streams = [cuda.stream() for _ in range(self.n_streams)]

        if self.graphs is None:
            self.models = [GPUModel(n_agents=n_agents, stream=stream, **kwargs) for stream in self.streams]
        else:
            self.models = [GPUModel(n_agents=n_agents, stream=stream, G=self.graphs[i], **kwargs) for i, stream in enumerate(self.streams)]

        self.rng_states = [create_xoroshiro128p_states(n_agents, seed=random_seeds[i]) for i in range(self.n_streams)]

    def MCS(self, steps: int, verbose: bool = False) -> None:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(model.MCS, steps, self.tpb, self.bpg, rng_state) for model, rng_state in zip(self.models, self.rng_states)]
            for future in tqdm(futures):
                future.result()

    def save_wealths(self, filepath: str = None) -> None:
        if filepath is None:
            raise ValueError("Insert a valid filepath with the structure 'path/name'")
        wealths = np.array([[self.models[i].w] for i in range(self.n_streams)])
        np.save(filepath, wealths)

    def get_gini(self) -> tuple:
        ginis = [measures.gini(model.w) for model in self.models]
        return np.mean(ginis), np.std(ginis)

    def get_n_active(self) -> tuple:
        n_active = [measures.num_actives(model.w, model.w_min) for model in self.models]
        return np.mean(n_active), np.std(n_active)

    def get_n_frozen(self) -> tuple:
        n_frozen = [measures.num_frozen(model.w, model.w_min, model.G) for model in self.models]
        return np.mean(n_frozen), np.std(n_frozen)



# Example usage:
if __name__ == "__main__":
    model_params = {
        'n_agents': 1000,
        'f': 0.1,
        'measure_every': 10,
        'upd_w_every': 50,
        'upd_graph_every': 50
    }