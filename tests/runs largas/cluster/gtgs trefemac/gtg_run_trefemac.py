import numpy as np
from matplotlib import pyplot as plt
import os
import enzope
import networkx as nx
from tqdm import tqdm

def r1(x):
    return x**(-1)

def print_info(gtgs):
    cons = [gtgs[i].get_mean_degree()[0] for i in range(len(gtgs))]
    clust = [[nx.average_clustering(gtgs[i].G) for i in range(len(gtgs))]]
    assort = [nx.degree_assortativity_coefficient(gtgs[i].G) for i in range(len(gtgs))]
    print(f"Mean k: {np.mean(cons):.2f} +- {np.std(cons):.2f}")
    print(f"Mean clustering: {np.mean(clust):.2f} +- {np.std(clust):.2f}")
    print(f"Mean assortativity: {np.mean(assort):.2f} +- {np.std(assort):.2f}")

# Params
reps = 10
n_nodes = 1000
theta = 65
mcs = 200000

def ba_ensemble():
    # BA graphs
    # <k>: 3.992 +- 0.0000000000
    # C: 0.026 +- 0.005
    # r: -0.092 +- 0.011
    return [nx.barabasi_albert_graph(n_nodes, 2) for _ in range(reps)]

def er_ensemble():
    # ER graphs
    # <k>: 3.992 +- 0.0000000000
    # C: 0.003 +- 0.000
    # r: -0.001 +- 0.000
    return [nx.erdos_renyi_graph(n_nodes, 0.00405) for _ in range(reps)]

def graph_ensemble():
    gtgs = [enzope.graphs.graph_class.GTG(n_nodes=n_nodes, theta=theta, join='add', p_dist=r1) for i in range(reps)]
    return gtgs


# Esto esta copiado con distintos f_set porque el cluster me reta que uso muchos recursos
f_set = np.arange(0, .25, 0.02)

for i, f in f_set:
    graphs = ba_ensemble()
    
    # Esto es un fix para que corra el programa, ya que la funcion get_neighbours_gpu solo esta en la clase GTG
    gtgs = [enzope.graphs.graph_class.GTG(n_nodes, theta=10000) for _ in range(reps)]
    for i in range(reps):
        gtgs[i].G = graphs[i]
        
    ensemble = enzope.GPUEnsemble(n_models=reps, n_agents=n_nodes, graphs=gtgs, f=f)

    for model, rng_state in tqdm(zip(ensemble.models, ensemble.rng_states)):
        model.MCS(mcs, ensemble.tpb, ensemble.bpg, rng_state)

    ginis = [enzope.measures.gini(model.w) for model in ensemble.models]
    print(f"{f}, {np.mean(ginis)}, {np.std(ginis)},")

    del graphs, ensemble


f_set = np.arange(.25, .502, 0.02)

for f in f_set:
    graphs = ba_ensemble()
    
    # Esto es un fix para que corra el programa, ya que la funcion get_neighbours_gpu solo esta en la clase GTG
    gtgs = [enzope.graphs.graph_class.GTG(n_nodes, theta=10000) for _ in range(reps)]
    for i in range(reps):
        gtgs[i].G = graphs[i]

    ensemble = enzope.GPUEnsemble(n_models=reps, n_agents=n_nodes, graphs=gtgs, f=f)

    for model, rng_state in tqdm(zip(ensemble.models, ensemble.rng_states)):
        model.MCS(mcs, ensemble.tpb, ensemble.bpg, rng_state)

    ginis = [enzope.measures.gini(model.w) for model in ensemble.models]
    print(f"{f}, {np.mean(ginis)}, {np.std(ginis)},")

    del graphs, ensemble