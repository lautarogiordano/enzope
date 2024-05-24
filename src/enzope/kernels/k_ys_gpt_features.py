# Agregados por gpt
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