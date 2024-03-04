import numpy as np
from enzope.graphs.graph_class import GTG
import enzope
import networkx as nx

def r1(x):
    return 1/x

def print_info(gtgs):
    cons = [gtgs[i].get_mean_connectivity() for i in range(len(gtgs))]
    clust = [[nx.average_clustering(gtgs[i].G) for i in range(len(gtgs))]]
    assort = [nx.degree_assortativity_coefficient(gtgs[i].G) for i in range(len(gtgs))]
    print(f"Mean k: {np.mean(cons):.2f} +- {np.std(cons):.2f}")
    print(f"Mean clustering: {np.mean(clust):.2f} +- {np.std(clust):.2f}")
    print(f"Mean assortativity: {np.mean(assort):.2f} +- {np.std(assort):.2f}")


a = 1.5
m = 1
n_systems_set = [200, 100, 50]
n_nodes_set = [1000, 2000, 5000]
f_set = [0, 0.1, 0.2, .25]
mcs = 50000
theta_set = [380, 380*1.8, 380*3.5]
w_min = 3e-17

for n_nodes, n_systems, theta in zip(n_nodes_set, n_systems_set, theta_set):
    weights = [dict(enumerate((np.random.default_rng().pareto(a, n_nodes) + 1) * m)) for _ in range(n_systems)]
    gtgs = [GTG(n_nodes=n_nodes, theta=theta, w0=weights[i], join='mul', p_dist=r1) for i in range(n_systems)]
    print("@@@@@@@@")
    print(f"n_nodes={n_nodes}, n_systems={n_systems}, theta={theta}, mcs={mcs}")
    print_info(gtgs)
    print("@@@@@@@@")
    for f in f_set:
        ensemble = enzope.models.model.GPUEnsemble(n_models=n_systems, n_agents=n_nodes, f=0, graphs=gtgs, w_min=w_min) 
        ensemble.MCS(mcs)
        ginis, std_gin = ensemble.get_gini()
        actives, std_active = ensemble.get_n_active()
        frozen, std_frozen = ensemble.get_n_frozen()

        print(f"f={f}")
        print(f"Gini: {ginis:.3f} +- {std_gin:.3f}")
        print(f"Actives: {actives:.3f} +- {std_active:.3f}")
        print(f"Frozen: {frozen:.3f} +- {std_frozen:.3f}")
        print("---------")

