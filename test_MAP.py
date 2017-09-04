import numpy as np
from training import training_MAP
import kernel
import ghat
import scipy
import time
from scipy.spatial.distance import pdist, squareform
from sklearn import preprocessing

from gamma_r import gamma_MAP
from sklearn.metrics.pairwise import rbf_kernel #remember
np.random.seed(0)

T = np.array([[1,1/2, 1/3,1/4,1/5,1/6,1/7,1/8,1/9],[0,1/2, 1/3,1/4,1/5,1/6,1/7,1/8,1/9],
              [0,0, 1/3,1/4,1/5,1/6,1/7,1/8,1/9], [0,0, 0,1/4,1/5,1/6,1/7,1/8,1/9], [0,0, 0,0,1/5,1/6,1/7,1/8,1/9],[0,0, 0,0,0,1/6,1/7,1/8,1/9],
              [0,0,0,0,0,0,1/7,1/8,1/9],[0,0,0,0,0,0,0,1/8,1/9],[0,0,0,0,0,0,0,0,1/9]])

train = np.load("Web30KModified/small_max_zeros_data_train.npy")
print(train.shape)
train_r = np.load("Web30KModified/small_max_zeros_rel_train.npy")
print(train_r.shape)

val = np.load("Web30KModified/small_max_zeros_data_val.npy")
print(val.shape)
val_r = np.load("Web30KModified/small_max_zeros_rel_val.npy")

test = np.load("Web30KModified/small_max_zeros_data_test.npy")
print(test.shape)
test_r = np.load("Web30KModified/small_max_zeros_rel_test.npy")

perm_matrices = list(np.load('Web30KModified/perm_matrices_9.npy'))

def objective_fun(W, T, P):
    return np.trace(np.transpose(P).dot(W).dot(P).dot(T))

def train_max(W,T,perm_matrices):
    P_best = 0
    value = 0
    for i,P in enumerate(perm_matrices):
        value_new = objective_fun(W,T,P)
        if value_new > value:
            value = value_new
            P_best = P
    return value, P_best

def g_MAP(train, Kx, K_inv, gamma_train):
    """
    Gives a prediction for a specic input x, and a training set, with linear kernel and bias C.
    :param train_r:
    :param x:
    :param K_inv:
    :param gamma_train:
    :param c:
    :return:
    """
    nr_queries = train.shape[0]
    prediction = ghat.g_hat(nr_queries, K_inv, Kx, gamma_train)
    return prediction

def test_proc(train,test, test_r, K_inv, gamma_train, perm_matrices, c,T):
    """
    Tests on all test data points for a bias c, and training input K_inv, gamma_train (trained with l)
    :param train_r: matrix of training rel labels, padded with zeros np_array (nr_queries x nr_documents x dim_feature)
    :param test: matrix test data, padded with zeros np_array (nr_queries x nr_documents x dim_feature)
    :param test_r: matrix test rel labels , padded with zeros np_array (nr_queries x nr_documents x dim_feature)
    :param K_inv: (K + nlI)^{-1} (nr_documents x nr_documents)
    :param gamma_train: array of all gamma(r^{i}) (nr_queries x nr_documents x nr_documents)
    :param perm_matrices: All possible permutation matrices with dim nr_documents
    :param c: bias for linear kernel
    :param T: Upper triangle matrix defined for the MAP
    :return: Returns the mean average precision.
    """
    nr_test_queries = test_r.shape[0]
    shape_1 = train.shape

    AP = 0

    j = 0
    for i in range(nr_test_queries):
        j+= 1
        x = np.expand_dims(test[i, :, :], axis=0)
        shape_x = x.shape
        x = np.reshape(x, (shape_x[0], shape_x[1]*shape_x[2]))
        train_reshape = np.reshape(train, (shape_1[0], shape_1[1]*shape_1[2]))
        Kx = kernel.rbf_kernel(train_reshape, x,c)

        prediction = g_MAP(train, Kx, K_inv, gamma_train)

        (value,optimal_perm) = train_max(prediction,T,perm_matrices)
        r = gamma_MAP(test_r[i, :], k=10)
        r[r>1] == 1
        AP = AP * (j-1)/j + objective_fun(r,T,optimal_perm)/j

    return AP


l = 0.006
c = 0.0005
train = np.concatenate((train, val), axis = 0) #FOR TESTING, thus uncomment for validating!
train_r = np.concatenate((train_r, val_r), axis = 0)

shape_train = train.shape #FOR TRAINING
train = np.reshape(train, (shape_train[0]*shape_train[1], shape_train[2]))
scaler = preprocessing.StandardScaler().fit(train)
train = scaler.transform(train)
train = np.reshape(train, (shape_train[0],shape_train[1], shape_train[2]))
train_reshape = np.reshape(train, (shape_train[0],shape_train[1]*shape_train[2]))

#shape_val = val.shape #FOR TRAINING
#val = np.reshape(val, (shape_val[0]*shape_val[1], shape_val[2]))
#val = scaler.transform(val)
#val= np.reshape(val, (shape_val[0],shape_val[1], shape_val[2]))

shape_test = test.shape# FOR TESTING
test = np.reshape(test, (shape_test[0]*shape_test[1], shape_test[2]))
test = scaler.transform(test)
test = np.reshape(test, (shape_test[0],shape_test[1], shape_test[2]))

#pairwise_dists = squareform(pdist(train_reshape, 'euclidean'))
#print(np.median(pairwise_dists)) #median of distances gamma =
K = kernel.rbf_kernel(train_reshape, gamma = c) #need at least 7 (1e-8) zeros. kernel.
K_inv, gamma_train = training_MAP(train, train_r, K, l)
test_proc(train,test,test_r, K_inv, gamma_train, perm_matrices, c,T)

def validation(train, train_r, val, val_r,test,test_r, perm_matrices, T):
    hyperparameters_c = [0.00009,0.0004, 0.0005, 0.0006,0.00077,0.0008,0.0009, 0.001]
    hyperparameters_l = [0.005, 0.006, 0.007, 0.008, 0.01,0.02,0.03, 0.05, 0.06,0.07, 0.08]
    shape_1 = train.shape

    best_AP = 0
    for c in hyperparameters_c:
        start_time = time.time()
        for l in hyperparameters_l:
            train_reshape = np.reshape(train, (shape_1[0], shape_1[1] * shape_1[2]))
            K = kernel.rbf_kernel(train_reshape, train_reshape,c) #KERNEL, perhaps reshape, check before validation
            K_inv, gamma_train = training_MAP(train, train_r, K, l)  #train with l
            AP = test_proc(train, val, val_r, K_inv, gamma_train, perm_matrices, c,T) #test on validation set for trained with l and c

            if AP > best_AP:
                best_AP = AP
                l_opt = l
                c_opt = c
                print(best_AP, 'best_AP_so far', c_opt, l_opt)
        end_time = time.time()
        print("total time taken this loop: ", end_time - start_time)

    print(l_opt, c_opt)

#validation(train, train_r, val, val_r, test, test_r,perm_matrices,T)#optimal so far: 0.01, 0.00077/0.006, 0.0005


