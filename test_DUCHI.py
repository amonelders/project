import kernel
import ghat
import igraph
import pandas as pd
import numpy as np
from gamma_r import gamma_DUCHI

def adj_to_graph(Adj):
    a = pd.DataFrame(Adj)
    A = a.values

    g = igraph.Graph.Adjacency((A > 0).tolist())
    g.es['weight'] = A[A.nonzero()]
    g.vs['label'] = a.columns
    return g

A = np.array([[0,1,0], [0,0,2], [3,0,0]])
#print(adj_to_graph(A))

def test_DUCHI(ghat):
    ggraph = adj_to_graph(ghat)
    fas = igraph.Graph.feedback_arc_set(ggraph, weights = ggraph.es['weight'], method='eades')
    ggraph.delete_edges(fas)
    return ggraph.topological_sorting()

#r = np.array([1,0,2,0])
#y=list(np.array(test_DUCHI(np.outer(r,r))) + 1)


def loss_DUCHI(y,r):
    nr_documents = r.shape[0]
    loss = 0
    edges = gamma_DUCHI(r,10)
    print(edges)
    for edge, weight in np.ndenumerate(edges):
        if y[edge[0]] - y[edge[1]] > 0: #note, because this is the ranking, 0 is the better ranking. Seems to work for small example!.
            loss += weight
    return 1/nr_documents*loss
