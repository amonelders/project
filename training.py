import ghat
import kernel
import numpy as np
import gamma_r
from scipy import linalg

def training_NDCG_rbf(train_r,K,l,k=10):
    """dont forget kernel
    :param train:
    :param train_r:
    :param l:
    :param k:
    :return:
    """
    n = train_r.shape[0]
    K_inv = linalg.inv((K + n * l * np.identity(n)))
    gamma_tr = ghat.gamma_train(train_r,gamma_r.gamma_NDCG,k)
    return K_inv, gamma_tr

def training_NDCG(train_r,K,l,k=10):
    """dont forget kernel
    :param train:
    :param train_r:
    :param l:
    :param k:
    :return:
    """
    n = train_r.shape[0]
    K_inv = linalg.inv((K + n * l * np.identity(n)))
    gamma_tr = ghat.gamma_train(train_r,gamma_r.gamma_NDCG,k)
    return K_inv, gamma_tr

def training_DUCHI(train, train_r, l):
    k = train_r.shape[1]
    K_inv = ghat.train_kernel_inv(l, kernel.kernel_DUCHI, train)
    gamma_tr = ghat.gamma_train(train_r, gamma_r.gamma_DUCHI, k)
    return K_inv, gamma_tr

def training_MAP(train, train_r, K, l):
    """
    :param train:
    :param train_r:
    :param K:
    :param l:
    :return:
    """
    k = train_r.shape[1]
    K_inv = ghat.train_kernel_inv(train,K,l)
    gamma_tr = ghat.gamma_train(train_r, gamma_r.gamma_MAP, k)
    return K_inv, gamma_tr

def training_MAP_UB_DUCHI(train, train_r, K, l):
    """
    :param train:
    :param train_r:
    :param K:
    :param l:
    :return:
    """
    k = train_r.shape[1]
    K_inv = ghat.train_kernel_inv(train,K,l)
    gamma_tr = ghat.gamma_train(train_r, gamma_r.gamma_MAP_UB_DUCHI, k)
    return K_inv, gamma_tr