import igraph
import pandas as pd

def adj_to_graph(Adj):
    a = pd.DataFrame(Adj)
    A = a.values

    g = igraph.Graph.Adjacency((A > 0).tolist())
    g.es['weight'] = A[A.nonzero()]
    g.vs['label'] = a.columns
    return g
