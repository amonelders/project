import igraph
import numpy as np
import pandas as pd

def adj_to_graph(Adj):
    a = pd.DataFrame(Adj)
    A = a.values

    g = igraph.Graph.Adjacency((A > 0).tolist())
    g.es['weight'] = A[A.nonzero()]
    g.vs['label'] = a.columns
    return g

adj = 2-np.array([[0,2], [1,0]])
g = adj_to_graph(adj)

fas = igraph.Graph.feedback_arc_set(g,g.es['weight'],method = 'eades')
g.delete_edges(fas)
print(g.topological_sorting())


#remember, we have to minimize the duchi loss, so we don't have to multiply ghat by -1.

#seems like we have a maximum weight feedback arc-set problem. amax (+1)? - prediction again?
#todo, remove edges, from directed graph to ordering, calculate loss.
# is it r[i] - r[j]? or r[j] - r[i]?


"""
node_names = ['1','2', '3']
a = pd.DataFrame([[0,0,1], [0,0,1], [1,1,0]])
A = a.values

g = igraph.Graph.Adjacency((A>0).tolist())

g.es['weight'] = Ah[A.nonzero()]
g.vs['label'] = node_names
igraph.plot(g, layout="rt", labels=True, margin=80)

df_from_g = pd.DataFrame(g.get_adjacency(attribute='weight').data,
                         columns=g.vs['label'], index=g.vs['label'])
(df_from_g == a).all().all()
"""