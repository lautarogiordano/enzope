import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython.display import display

from .custom_gtg import geographical_threshold_graph_custom


def r1(x):
    return x ** (-1)


# ---------------------- #
#        Classes         #
# ---------------------- #


class BaseGraph:
    """
    Base class for representing a graph.

    Args:
        n_nodes (int): The number of nodes in the graph.
        figpath (str, optional): The path to save the figures. Defaults to the current working directory.
        plotable (bool, optional): Flag indicating whether the graph is plotable. Defaults to False.

    Attributes:
        n_nodes (int): The number of nodes in the graph.
        G (networkx.Graph): The graph object.
        figpath (str): The path to save the figures.
        temppath (str): The path to the temporary directory.
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes.Axes): The axes object.

    """

    def __init__(
        self,
        n_nodes,
        figpath=os.getcwd(),
        plotable=False,
    ):
        self.n_nodes = n_nodes
        self.G = nx.empty_graph(n_nodes)

        if plotable:
            self.fig, self.ax = plt.subplots(dpi=150)
            self.ax.set_xlim(-0.03, 1.03)
            self.ax.set_ylim(-0.03, 1.03)
            plt.close(self.fig)

        if figpath is not None:
            self.figpath = figpath
            self.temppath = os.path.join(self.figpath, "temp")

    # Some networkx functions for comfort
    def get_mean_degree(self):
        """
        Compute the mean degree of the graph.

        Returns:
            tuple: A tuple containing the mean degree and standard deviation of the degrees.

        """
        degrees = [d for n, d in self.G.degree()]
        return np.mean(degrees), np.std(degrees)

    def get_average_distance(self):
        """
        Compute the average shortest path length of the graph.

        Returns:
            float: The average shortest path length.

        """
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
        G0 = self.G.subgraph(Gcc[0])
        return nx.average_shortest_path_length(G0)

    def get_clustering(self):
        """
        Compute the average clustering coefficient of the graph.

        Returns:
            float: The average clustering coefficient.

        """
        return nx.average_clustering(self.G)

    def get_assortativity(self):
        """
        Compute the degree assortativity coefficient of the graph.

        Returns:
            float: The degree assortativity coefficient.

        """
        return nx.degree_assortativity_coefficient(self.G)


class GTG(BaseGraph):
    """
    Class representing a GTG (Geographical Threshold Graph).

    Parameters:
    - n_nodes (int): Number of nodes in the graph.
    - theta (float): Threshold value for edge creation.
    - metric (str, optional): Distance metric for edge creation. Default is Euclidean metric.
    - join (str, optional): Method for joining edges. Default is "add", which means additive rule.
    - w0 (float, optional): Initial weight for nodes. Default is taken from exponential distribution with mean 1.
    - posi (list, optional): List of node positions. Default is uniform random positions in [0,1]^d.
    - seed (int, optional): Seed value for random number generation. Default is None.
    - p_dist (float, optional): Distance rule for connecting the nodes. Default is r**-2.
    - **kwargs: Additional keyword arguments.

    Attributes:
    - theta (float): Threshold value for edge creation.
    - metric (str): Distance metric for edge creation.
    - posi (list): List of node positions.
    - join (str): Method for joining edges.
    - seed (int): Seed value for random number generation.
    - G (networkx.Graph): Graph object representing the GTG.
    - w (dict): Dictionary of node weights.

    Methods:
    - __call__(self, *args, **kwds): Returns the graph object.
    - plot_snapshot(self, w_min=1e-17, new_w=None, mcs=None, mode="show", *args, **kwargs): Plots a snapshot of the graph.
    - update_weights(self, new_weights, *args, **kwargs): Updates the node weights.
    - update_graph(self, *args, **kwargs): Updates the graph object.
    - get_opponents_cpu(self, *args, **kwargs): Returns an array of randomly chosen opponents for each node.
    - get_neighbors_array_gpu(self, *args, **kwargs): Returns arrays of cumulative neighbors and individual neighbors for each node (Used only for GPU runs).
    """

    def __init__(
        self,
        n_nodes,
        theta,
        metric=None,
        join="add",
        w0=None,
        posi=None,
        seed=None,
        p_dist=None,
        **kwargs,
    ):
        super().__init__(n_nodes, **kwargs)
        self.theta = theta
        self.metric = metric
        self.posi = posi
        self.join = join
        self.seed = seed

        self.G = geographical_threshold_graph_custom(
            n_nodes,
            theta,
            dim=2,
            pos=posi,
            weight=w0,
            metric=metric,
            p_dist=p_dist,
            seed=seed,
            join=join,
        )

        if self.posi is None:
            self.posi = [self.G.nodes[i]["pos"] for i in range(n_nodes)]

        self.w = dict(enumerate(self.G.nodes[i]["weight"] for i in range(self.n_nodes)))

    def __call__(self, *args, **kwds):
        return self.G

    # Modificar la funcion para que o bien guarde en temp o bien muestre en pantalla
    def plot_snapshot(
        self, w_min=1e-17, new_w=None, mcs=None, mode="show", *args, **kwargs
    ):
        """
        Plot a snapshot of the graph.

        Args:
            w_min (float, optional): Minimum weight threshold for determining dead nodes. Defaults to 1e-17.
            new_w (dict, optional): New weights to update the graph. Defaults to None.
            mcs (int, optional): Monte Carlo step for title plotting. Defaults to None.
            mode (str, optional): Plotting mode. Can be "show" or "save". Defaults to "show".
            *args, **kwargs: Additional arguments.

        Returns:
            None
        """
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
            pos=self.posi if self.posi is not None else nx.spring_layout(self.G),
            node_color=node_colors,
            edge_color=edge_colors,
            ax=self.ax,
        )

        if mode == "save":
            if mcs:
                filename = os.path.join(self.temppath, "test_{:05d}.png".format(mcs))
            else:
                filename = os.path.join(self.temppath, "test.png")
            self.fig.savefig(filename, format="PNG")

        if mode == "show":
            display(self.fig)

    def update_weights(self, new_weights, *args, **kwargs):
        """
        Updates the node weights.

        Parameters:
        - new_weights (list): List of new node weights.
        - *args, **kwargs: Additional arguments.

        Returns:
        - None
        """
        self.w = dict(enumerate(new_weights))
        nx.set_node_attributes(self.G, self.w, "weight")

    def update_graph(self, *args, **kwargs):
        """
        Updates the graph object.

        Parameters:
        - *args, **kwargs: Additional arguments.

        Returns:
        - None
        """
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
        """
        Returns an array of random opponents for each node.

        Parameters:
        - *args, **kwargs: Additional arguments.

        Returns:
        - opponents (numpy.ndarray): Array of opponents for each node.
        """
        opponents = np.full(self.n_nodes, fill_value=-1)
        for i in range(self.n_nodes):
            if neighbors := list(nx.all_neighbors(self.G, i)):
                opponents[i] = np.random.choice(neighbors)

        return opponents

    # Function inspired by cuTradeNet's toLL() function in:
    # https://github.com/Qsanti/cuTradeNet/blob/e20f29e65bcac448d7fcd17fac45746f90e8538e/cuTradeNet/Models/Utils/GraphManager.py
    def get_neighbors_array_gpu(self, *args, **kwargs):
        """
        Returns arrays of cumulative neighbors and individual neighbors for each node.

        Parameters:
        - *args, **kwargs: Additional arguments.

        Returns:
        - c_neighs (numpy.ndarray): Array of cumulative neighbors.
        - neighs (numpy.ndarray): Array of individual neighbors.
        """
        neighs = [(list(self.G.neighbors(i))) for i in range(self.n_nodes)]
        n_neighs = [0] + [len(x) for x in neighs]
        n_neighs = np.array(n_neighs, dtype=np.int32)
        # c_neighs has N+1 components, first is 0.
        c_neighs = np.cumsum(n_neighs)
        neighs = np.hstack(neighs).astype(np.int32)
        return c_neighs, neighs


# Acá iría la clase del multigraph


# ---------------------- #
#       Funciones        #
# ---------------------- #
