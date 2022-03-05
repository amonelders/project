#testing
import numpy as np
from scipy import linalg


def train_kernel_inv(train, K, l):
    """
    :param train: nr_data_tr x nr_documents x dim_feature_size (array)
    :param l: regularization parameter, float
    :return: K_inv, nr_data_tr x nr_data_tr (array, floats).
    """
    n = train.shape[0]
    K_inv = linalg.inv((K + n * l * np.identity(n)))
    return K_inv

def gamma_train(train_r,gamma,k):
    """
    :param train_r: nr_queries x nr_documents (array)
    :param k: ndcg at k (int)
    :return: nr_queries x k x nr_documents
    """
    nr_queries = train_r.shape[0]
    nr_documents = train_r.shape[1]

    gamma_tr = np.zeros(shape = (nr_queries, k, nr_documents))
    for i in range(nr_queries):
        gamma_tr[i, :, :] = gamma(train_r[i,:], k)
    return gamma_tr

def g_hat(nr_queries, K_inv, Kx, gamma_tr):
    """
    :param data_tr: nr_data_tr x nr_documents x dim_feature_size (array)
    :param K_inv:
    :param x: 1xnr_documents x dim_feature_size (needs to be 1!)
    :param gammas: all gamma(r) for training from gamma_train function
    :return: k x documents prediction
    """
    prediction = 0
    alphas = K_inv.dot(Kx)

    for i in range(nr_queries):
        prediction += alphas[i]*gamma_tr[i, :, :]

    prediction = np.amax(prediction) + prediction
    return prediction
