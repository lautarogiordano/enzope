from .custom_gtg import geographical_threshold_graph_custom
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import os


# ---------------------- #
#        Classes         #
# ---------------------- #


class BaseGraph:
    def __init__(
        self,
        n_nodes,
        upd_w=np.inf,
        upd_graph=np.inf,
        upd_plot=np.inf,
        figpath=None,
        plotable=False,
    ):
        self.n_nodes = n_nodes
        self.upd_w = upd_w
        self.upd_graph = upd_graph
        self.upd_plot = upd_plot
        self.G = None

        if plotable:
            self.fig, self.ax = plt.subplots(dpi=150)
            self.ax.set_xlim([-0.03, 1.03])
            self.ax.set_ylim([-0.03, 1.03])
            plt.close(self.fig)

        if figpath is not None:
            self.fighpath = figpath
            self.temppath = os.path.join(self.figpath, "temp")

    def plot_snapshot(self, *args, **kwargs):
        pass

    def update_weights(self, *args, **kwargs):
        pass

    def update_graph(self, *args, **kwargs):
        pass

    def get_opponents_cpu(self, *args, **kwargs):
        pass
    
    def get_mean_connectivity(self):
        pass
    
    def get_average_distance(self):
        pass

class GTG(BaseGraph):
    def __init__(
        self, n_nodes, theta, join="add", w0=None, posi=None, seed=None, **kwargs
    ):
        super().__init__(n_nodes, **kwargs)
        self.theta = theta
        self.posi = posi
        self.join = join
        self.seed = seed

        self.G = geographical_threshold_graph_custom(
            n_nodes,
            theta,
            dim=2,
            pos=posi,
            weight=w0,
            metric=None,
            p_dist=None,
            seed=seed,
            join=join,
        )

        if self.posi is None:
            self.posi = [self.G.nodes[i]["pos"] for i in range(n_nodes)]

        self.w = dict(enumerate(self.G.nodes[i]["weight"] for i in range(self.n_nodes)))

    def __call__(self, *args, **kwds):
        return self.G

    # Modificar la funcion para que o bien guarde en temp o bien muestre en pantalla
    def plot_snapshot(self, w_min=1e-17, new_w=None, mcs=None, mode="show", *args, **kwargs):
        # Esto se tiene que escribir mejor
        if new_w is not None:
            self.update_weights(new_w)

        a = np.array(list(self.w.values()))

        dead_nodes = [node for node, weight in self.w.items() if weight < w_min]

        node_size = 25 * (a)
        node_colors = ["r" if n in dead_nodes else "royalblue" for n in self.G.nodes]
        edge_colors = [
            "r" if (e[0] in dead_nodes or e[1] in dead_nodes) else "black"
            for e in self.G.edges
        ]

        self.ax.clear()
        if mcs is not None:
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

        if mode == "save":
            filename = os.path.join(self.temppath, "test_{:05d}.png".format(mcs))
            self.fig.savefig(filename, format="PNG")

        if mode == "show":
            display(self.fig)

    def update_weights(self, wealths, *args, **kwargs):
        self.w = dict(enumerate(wealths))
        nx.set_node_attributes(self.G, self.w, "weight")

    def update_graph(self, *args, **kwargs):
        self.G = geographical_threshold_graph_custom(
            self.n_nodes,
            self.theta,
            dim=2,
            pos=self.posi,
            weight=self.w,
            metric=None,
            p_dist=None,
            seed=self.seed,
            join=self.join,
        )

    def get_opponents_cpu(self, *args, **kwargs):
        opponents = np.full(self.n_nodes, fill_value=-1)
        for i in range(self.n_nodes):
            if neighbors := list(nx.all_neighbors(self.G, i)):
                opponents[i] = np.random.choice(neighbors)

        return opponents

    # Function inspired by cuTradeNet's toLL() function in:
    # https://github.com/Qsanti/cuTradeNet/blob/e20f29e65bcac448d7fcd17fac45746f90e8538e/cuTradeNet/Models/Utils/GraphManager.py
    def get_neighbors_array_gpu(self, *args, **kwargs):
        """ """
        neighs = [(list(self.G.neighbors(i))) for i in range(self.n_nodes)]
        n_neighs = [0] + [len(x) for x in neighs]
        n_neighs = np.array(n_neighs, dtype=np.int32)
        # c_neighs has N+1 components, first is 0.
        c_neighs = np.cumsum(n_neighs)
        neighs = np.hstack(neighs).astype(np.int32)
        return c_neighs, neighs
    
    # Some networkx functions for comfort
    def get_mean_connectivity(self):
        return np.mean(list(dict(nx.degree(self.G)).values()))
    
    def get_average_distance(self):
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
        G0 = self.G.subgraph(Gcc[0])
        return nx.average_shortest_path_length(G0)


# Acá iría la clase del multigraph


# ---------------------- #
#       Funciones        #
# ---------------------- #
