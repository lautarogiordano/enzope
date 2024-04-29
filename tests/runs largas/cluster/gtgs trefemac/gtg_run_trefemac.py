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
reps = 100
n_nodes = 1000
theta = 65
mcs = 200000

def graph_ensemble():
    gtgs = [enzope.graphs.graph_class.GTG(n_nodes=n_nodes, theta=theta, join='add', p_dist=r1) for i in range(reps)]
    # print(f'theta={theta}')
    # print_info(gtgs)
    return gtgs

f_set = np.arange(0, .501, 0.01)

for f in f_set:
    graphs = graph_ensemble()
    ensemble = enzope.GPUEnsemble(n_models=reps, n_agents=n_nodes, graphs=graphs, f=f)

    for model, rng_state in tqdm(zip(ensemble.models, ensemble.rng_states)):
        model.MCS(mcs, ensemble.tpb, ensemble.bpg, rng_state)

    ginis = [enzope.measures.gini(model.w) for model in ensemble.models]
    print(f"{f}, {np.mean(ginis)}, {np.std(ginis)},")

    del graphs, ensemble