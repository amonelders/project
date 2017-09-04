import numpy as np
import ghat
from scipy.optimize import linear_sum_assignment
import kernel
import training as tr
from gamma_r import gamma_NDCG
import time

train = np.load("Web10kModified" +'/' + '{}'.format(1)+'data_matrix_train.npy')
train_r = np.load("Web10kModified" + '/' + '{}'.format(1) + 'data_matrix_train_r.npy')
val = np.load("Web10kModified" + '/' + '{}'.format(1) + 'data_matrix_val.npy')
val_r = np.load("Web10kModified" + '/' + '{}'.format(1) + 'data_matrix_val_r.npy')

def test_NDCG_rbf(nr_queries, Kx, K_inv, gamma_train):
    prediction = ghat.g_hat(nr_queries,K_inv, Kx, gamma_train)
    return prediction

def test_ndcg(nr_queries, Kx, K_inv, gamma_train):
    prediction = ghat.g_hat(nr_queries,K_inv, Kx, gamma_train)
    return prediction

#FOR VALIDATION.
train_sh = train.shape
shape_train = train.shape


def test_proc(train, test, test_r, K_inv, gamma_train, c):
    nr_test_queries = test_r.shape[0]
    shape_train = train.shape
    ndcg = 0

    j = 0
    for i in range(nr_test_queries):
        j += 1
        x = np.expand_dims(test[i, :, :], axis=0)
        shape_x = x.shape
        x = np.reshape(x, (shape_x[0], shape_x[1] * shape_x[2]))
        train_reshape = np.reshape(train, (shape_train[0], shape_train[1] * shape_train[2]))
        Kx = kernel.lin_kernel(train_reshape, x, c)

        prediction = test_ndcg(shape_train[0], Kx, K_inv, gamma_train)
        row_ind, col_ind = linear_sum_assignment(prediction)

        r = gamma_NDCG(test_r[i, :], k=10)
        ndcg = ndcg * (j - 1) / j + r[row_ind, col_ind].sum() / j
        print(ndcg)
    return ndcg


def validation(train, train_r, val, val_r):
    hyperparameters_c = [0,1, 2, -1,-2,5,-5, 10]
    hyperparameters_l = [10e-6, 10e-5, 10e-4, 10e-3, 10e-2,1,10]
    shape_1 = train.shape

    best_NDCG = 0
    for c in hyperparameters_c:
        start_time = time.time()
        for l in hyperparameters_l:
            train_reshape = np.reshape(train, (shape_1[0], shape_1[1] * shape_1[2]))

            K = kernel.lin_kernel(train_reshape, train_reshape,c)
            K_inv, gamma_train = tr.training_NDCG(train_r, K, l)

            NDCG = test_proc(train, val, val_r, K_inv, gamma_train, c) #gets the test score for this validation set
            if NDCG > best_NDCG:
                best_NDCG = NDCG
                l_opt = l
                c_opt = c
                print(best_NDCG, 'best_NDCG_so far', c_opt, l_opt)
        end_time = time.time()
        print("total time taken this loop: ", end_time - start_time)

    print(l_opt, c_opt)

