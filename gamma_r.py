import numpy as np
from NDCG import dcg_score

def gamma_NDCG(r, k):
    """
    :param max_r: maximum relevance label)int) (will need to take the max of whole dataset, probably 5)
    :param r: 1 x nr_documents (array)
    :return: cost matrix for a particular query with relevance labels r, kxnr_documents (array)
    """
    nr_documents = r.shape[0]
    gamma_r = np.zeros(shape = (k, nr_documents))
    best = dcg_score(r,r,k)

    if best == 0:
        return gamma_r

    else:
        for j in range(k):
            discount = 1/np.log2(1+(j+1))
            gamma_r[j, :] = discount*(np.exp2(r) - 1)

    return (1/best)*gamma_r

def gamma_DUCHI(r,k):
    """
    :param r: relevance label for a set of documents
    :return: nr_documents x nr_documents sized adjacency matrix
    """
    nr_documents = r.shape[0]
    gamma_r = np.zeros(shape = (nr_documents, nr_documents))
    print(r)
    for i,ri in enumerate(r):
        for j,rj in enumerate(r):
            diff = r[i] - [rj]
            print(diff)
            if diff > 0:
                gamma_r[i,j] = diff
    return gamma_r

def gamma_MAP(r,k):
    """
    :param r: relevance label for a set of documents
    :return: nr_documents x nr_documents sized adjacency matrix
    """
    alpha_r = np.sum(r)
    if alpha_r == 0:
        return np.zeros(shape=(r.shape[0], r.shape[0]))
    return 1/alpha_r*np.outer(r,r)
