import math
from itertools import combinations

import networkx as nx
from networkx.utils import py_random_state


# Generating function for additive and multiplicative threshold graphs,
# slightly modified version from the function provided in the networkx package.
@py_random_state(7)
def geographical_threshold_graph_custom(
    n,
    theta,
    dim=2,
    pos=None,
    weight=None,
    metric=None,
    p_dist=None,
    seed=None,
    join="add",
):
    r"""Returns a geographical threshold graph.

    The geographical threshold graph model places $n$ nodes uniformly at
    random in a rectangular domain.  Each node $u$ is assigned a weight
    $w_u$. Two nodes $u$ and $v$ are joined by an edge if

    .. math::

       - Additive type:
       (w_u + w_v)p_{dist}(r) \ge \theta

       - Multiplicative type
       (w_u * w_v)p_{dist}(r) \ge \theta

    where `r` is the distance between `u` and `v`, `p_dist` is any function of
    `r`, and :math:`\theta` as the threshold parameter. `p_dist` is used to
    give weight to the distance between nodes when deciding whether or not
    they should be connected. The larger `p_dist` is, the more prone nodes
    separated by `r` are to be connected, and vice versa.

    Parameters
    ----------
    n : int or iterable
        Number of nodes or iterable of nodes
    theta: float
        Threshold value
    dim : int, optional
        Dimension of graph
    pos : dict
        Node positions as a dictionary of tuples keyed by node.
    weight : dict
        Node weights as a dictionary of numbers keyed by node.
    metric : function
        A metric on vectors of numbers (represented as lists or
        tuples). This must be a function that accepts two lists (or
        tuples) as input and yields a number as output. The function
        must also satisfy the four requirements of a `metric`_.
        Specifically, if $d$ is the function and $x$, $y$,
        and $z$ are vectors in the graph, then $d$ must satisfy

        1. $d(x, y) \ge 0$,
        2. $d(x, y) = 0$ if and only if $x = y$,
        3. $d(x, y) = d(y, x)$,
        4. $d(x, z) \le d(x, y) + d(y, z)$.

        If this argument is not specified, the Euclidean distance metric is
        used.

        .. _metric: https://en.wikipedia.org/wiki/Metric_%28mathematics%29
    p_dist : function, optional
        Any function used to give weight to the distance between nodes when
        deciding whether or not they should be connected. `p_dist` was
        originally conceived as a probability density function giving the
        probability of connecting two nodes that are of metric distance `r`
        apart. The implementation here allows for more arbitrary definitions
        of `p_dist` that do not need to correspond to valid probability
        density functions. The :mod:`scipy.stats` package has many
        probability density functions implemented and tools for custom
        probability density definitions, and passing the ``.pdf`` method of
        scipy.stats distributions can be used here. If ``p_dist=None``
        (the default), the exponential function :math:`r^{-2}` is used.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    Graph
        A random geographic threshold graph, undirected and without
        self-loops.

        Each node has a node attribute ``pos`` that stores the
        position of that node in Euclidean space as provided by the
        ``pos`` keyword argument or, if ``pos`` was not provided, as
        generated by this function. Similarly, each node has a node
        attribute ``weight`` that stores the weight of that node as
        provided or as generated.

    Examples
    --------
    Specify an alternate distance metric using the ``metric`` keyword
    argument. For example, to use the `taxicab metric`_ instead of the
    default `Euclidean metric`_::

        >>> dist = lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
        >>> G = nx.geographical_threshold_graph(10, 0.1, metric=dist)

    .. _taxicab metric: https://en.wikipedia.org/wiki/Taxicab_geometry
    .. _Euclidean metric: https://en.wikipedia.org/wiki/Euclidean_distance

    Notes
    -----
    If weights are not specified they are assigned to nodes by drawing randomly
    from the exponential distribution with rate parameter $\lambda=1$.
    To specify weights from a different distribution, use the `weight` keyword
    argument::

    >>> import random
    >>> n = 20
    >>> w = {i: random.expovariate(5.0) for i in range(n)}
    >>> G = nx.geographical_threshold_graph(20, 50, weight=w)

    If node positions are not specified they are randomly assigned from the
    uniform distribution.

    References
    ----------
    .. [1] Masuda, N., Miwa, H., Konno, N.:
       Geographical threshold graphs with small-world and scale-free
       properties.
       Physical Review E 71, 036108 (2005)
    .. [2]  Milan Bradonjić, Aric Hagberg and Allon G. Percus,
       Giant component and connectivity in geographical threshold graphs,
       in Algorithms and Models for the Web-Graph (WAW 2007),
       Antony Bonato and Fan Chung (Eds), pp. 209--216, 2007
    """
    G = nx.empty_graph(n)
    # If no weights are provided, choose them from an exponential
    # distribution.
    if weight is None:
        weight = {v: seed.expovariate(1) for v in G}
    # If no positions are provided, choose uniformly random vectors in
    # Euclidean space of the specified dimension.
    if pos is None:
        pos = {v: [seed.random() for i in range(dim)] for v in G}
    # If no distance metric is provided, use Euclidean distance.
    if metric is None:
        metric = math.dist
    nx.set_node_attributes(G, weight, "weight")
    nx.set_node_attributes(G, pos, "pos")

    # if p_dist is not supplied, use default r^-2
    if p_dist is None:
        def p_dist(r):
            return r**-2

    # Returns ``True`` if and only if the nodes whose attributes are
    # ``du`` and ``dv`` should be joined, according to the threshold
    # condition.
    def should_join(pair):
        u, v = pair
        u_pos, v_pos = pos[u], pos[v]
        u_weight, v_weight = weight[u], weight[v]

        # Lautaro Modified. If not additive, weights join multiplicatively
        if join == "add":
            return (u_weight + v_weight) * p_dist(metric(u_pos, v_pos)) >= theta

        if join == "mul":
            return (u_weight * v_weight) * p_dist(metric(u_pos, v_pos)) >= theta

        else:
            raise TypeError("join should be 'add' or 'mul'.")

    G.add_edges_from(filter(should_join, combinations(G, 2)))
    return G
