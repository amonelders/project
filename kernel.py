import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import rbf_kernel #remember

def lin_kernel(queries_1, queries_2,c):
    """
    :param query_i: nr_of_queries_i x nr_documents x dim_feature_size (array) --> Turn into one large row vector
    :return: nr_of_queries_1 x nr_queries_2, array note nr_queries_trx1 for one test point, like we want.
    """
    return queries_1.dot(np.transpose(queries_2)) + c

def kernel_DUCHI(queries_1, queries_2, a, p):
    """
    need to check if correct .
    :param queries_1:
    :param queries_2:
    :return:
    """
    shape_1 = queries_1.shape
    shape_2 = queries_2.shape

    query_1 = np.reshape(queries_1, (shape_1[0], shape_1[1]*shape_1[2]))
    query_2 = np.reshape(queries_2, (shape_2[0], shape_2[1]*shape_2[2]))

    num = (1 + query_1.dot(np.transpose(query_2)))**p
    denom = np.expand_dims((a + np.sum(query_1**2,1))*(a+np.sum(query_2**2,1)), axis = 1)

    return num/denom**(p/2)

def sum_rbf_kernel(X, Y, gamma):
    #Not shaped.
    K = 0
    X_shape = X.shape
    for i in range(X_shape[1]):
        K += rbf_kernel(X[:,i,:], Y[:,i,:])
    return 1/X_shape[1]*K

#train = np.load("Web30KModified/small_max_zeros_data_train.npy")

#print(sum_rbf_kernel(train, train, gamma = 0.0001))




