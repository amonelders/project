import igraph
import numpy as np
from gamma_r import gamma_DUCHI
from adj_to_graph import adj_to_graph

def test_DUCHI(ghat):
    ggraph = adj_to_graph(ghat)
    fas = igraph.Graph.feedback_arc_set(ggraph, weights = ggraph.es['weight'], method='eades')
    ggraph.delete_edges(fas)
    return ggraph.topological_sorting()

def loss_DUCHI(y,r):
    nr_documents = r.shape[0]
    loss = 0
    edges = gamma_DUCHI(r,10)
    print(edges)
    for edge, weight in np.ndenumerate(edges):
        if y[edge[0]] - y[edge[1]] > 0:
            loss += weight
    return 1/nr_documents*loss
