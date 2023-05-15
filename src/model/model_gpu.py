import numpy as np
import cupy as cp


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
        self.every = save_every


class GPUModel(BaseModel):
    def __init__(self, w_0, **kwargs):  # sourcery skip: assign-if-exp
        super().__init__(**kwargs)
        # Initialize n agents with random risks and wealth between (0, 1]
        # and normalize wealth
        # n[i, 0] is the wealth and n[i, 1] is the risk of agent i
        self.w = cp.random.rand(self.n_agents).astype(cp.float32)
        self.r = cp.random.rand(self.n_agents).astype(cp.float32)
        if w_0 is not None:
            self.w = w_0
        else:
            self.w = self.w / (np.sum(self.w))
        # self.gini = [self.get_gini()]
        # self.n_active = [self.get_actives()]

    def MCS(self, steps):
        """
        Main MC loop
        """
        for mcs in range(steps):
            j = self.get_opponents()

            for i, ji in enumerate(j):
                # Check both agents have w > w_min and node is not isolated
                if self.is_valid(i, ji) and ji != -1:
                    dw = self.get_dw(i, ji)
                    winner = self.choose_winner(i, ji)
                    dw = np.where(winner == i, dw, -dw)
                    self.update_wealth(i, ji, dw)

            
            # After self.every MCS append new Gini index
            # if (mcs + 1) % self.every == 0:
                # self.gini.append(self.get_gini())
                # self.n_active.append(self.get_actives())

class GPUEnsemble:
    def __init__(self):
        


class CPUModel(BaseModel):
    def __init__(self, w_0, **kwargs):  # sourcery skip: assign-if-exp
        super().__init__(**kwargs)
        # Initialize n agents with random risks and wealth between (0, 1]
        # and normalize wealth
        # n[i, 0] is the wealth and n[i, 1] is the risk of agent i
        self.w = cp.random.rand(self.n_agents).astype(cp.float32)
        self.r = cp.random.rand(self.n_agents).astype(cp.float32)
        if w_0 is not None:
            self.w = w_0
        else:
            self.w = self.w / (np.sum(self.w))
        # self.gini = [self.get_gini()]
        # self.n_active = [self.get_actives()]

    def get_opponents(self):
        if self.G is None:
            random_array = np.random.randint(0, self.N, self.N)
            indices = np.arange(0, self.N)
            # Create array of random numbers that are not equal to the index
            # If i=j then assign j'=i+1 (j'=0 if i=N-1)
            random_array = np.where(
                random_array == indices, (random_array + 1) % self.N, random_array
            )
        else:
            random_array = np.full(self.N, fill_value=-1)
            for i in range(self.N):
                if neighbors := list(nx.all_neighbors(self.G, i)):
                    random_array[i] = np.random.choice(neighbors)

        return random_array

    def is_valid(self, i, j):
        # Check if both agents have w > w_min
        return (self.n[i, 0] > self.w_min) and (self.n[j, 0] > self.w_min)

    def get_dw(self, i, j):
        return np.minimum(self.n[i, 0] * self.n[i, 1], self.n[j, 0] * self.n[j, 1])

    def get_gini(self):
        w = np.sort(self.n[:, 0])
        p_cumsum = np.cumsum(w) / np.sum(w)
        B = np.sum(p_cumsum) / self.N
        return 1 + 1 / self.N - 2 * B

    def get_actives(self):
        return np.sum(self.n[:, 0] > self.w_min)

    def get_liquidity():
        return

    def update_wealth(self, i, j, dw):
        self.n[i, 0] += dw
        self.n[j, 0] -= dw

    def choose_winner(self, i, j):
        raise Exception("You need to choose a valid model.")

    def update_weights(self):
        w = dict(enumerate(self.n[:, 0]))
        nx.set_node_attributes(self.G, w, "weight")

    def plot_snapshot(self, mcs):
        ## Esto se tiene que borrar y escribir mejor
        w = dict(enumerate(self.n[:, 0]))
        a = np.array(list(w.values()))

        dead_nodes = [node for node, weight in w.items() if weight < self.w_min]

        node_size = 500 * np.sqrt(a)
        node_colors = plt.cm.coolwarm(100 * a)
        edge_colors = [
            "r" if (e[0] in dead_nodes or e[1] in dead_nodes) else "black"
            for e in self.G.edges
        ]

        filename = os.path.join(self.temppath, "test_{:05d}.png".format(mcs))

        self.ax.clear()
        self.ax.set_title(f"t = {mcs}")
        nx.draw(
            self.G,
            node_size=node_size,
            width=0.2,
            pos=self.posi,
            node_color=node_colors,
            edge_color=edge_colors,
            ax=self.ax,
        )
        self.fig.savefig(filename, format="PNG")

    def MCS(self, steps):
        """
        Main MC loop
        """
        for mcs in range(steps):
            if mcs % self.plot == 0:
                self.plot_snapshot(mcs)

            j = self.get_opponents()

            for i, ji in enumerate(j):
                # Check both agents have w > w_min and node is not isolated
                if self.is_valid(i, ji) and ji != -1:
                    dw = self.get_dw(i, ji)
                    winner = self.choose_winner(i, ji)
                    dw = np.where(winner == i, dw, -dw)
                    self.update_wealth(i, ji, dw)

            # After self.update_w update weights
            if mcs % self.update_w == 0:
                self.update_weights()

            # Recompute the links if the network is dynamic
            if (mcs + 1) % self.update_links == 0 and self.G is not None:
                self.G = nx.geographical_threshold_graph(
                    self.N,
                    theta=self.theta,
                    weight=self.n[:, 0],
                    dim=2,
                    pos=self.posi,
                    additive=self.additive,
                )
            # After self.every MCS append new Gini index
            if (mcs + 1) % self.every == 0:
                self.gini.append(self.get_gini())
                self.n_active.append(self.get_actives())
