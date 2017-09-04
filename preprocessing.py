import numpy as np
import os
import re
from itertools import islice
print(os.getcwd())

#dim = 46

def find_max_nr_doc(data):
    """
    finds the maximum nr of documents returned over all queries
    :param data:
    :param queries:
    :return:
    """
    queries = list(set(data[:, 1].astype(int)))
    max_nr = 0
    for query in queries:
        n_max = data[data[:,1] == query].shape[0]
        if n_max > max_nr:
            max_nr = n_max
    return max_nr

def find_nr_queries_max(data, max_nr_doc):
    query_max_nr_doc = []
    queries = list(set(data[:, 1].astype(int)))
    for query in queries:
        nr_doc = data[data[:, 1]==query].shape[0]
        if nr_doc < max_nr_doc:
            query_max_nr_doc.append(query)
    return query_max_nr_doc, max_nr_doc

def remove_large_query(queries_small,data):
    (tot_nr_doc, _) =  data.shape

    small_array = []
    for i in range(tot_nr_doc):
        if data[i,1] in queries_small:
            small_array.append(data[i,:])
    return np.array(small_array)

def add_zeros(data, dim_feature_size,type,max_nr = 9):
    """
    :param data: (nr_queries * nr_documents x dim_feature_size) np array
    :param queries: ordered list of unique queries
    :return: (nr_queries, max_nr, dim_feature_size) (documents), (nr_queries, max_nr) (rel_labels) np array
    """
    queries = list(set(data[:, 1].astype(int)))
    nr_queries = len(queries)
    print("Total nr of queries: {}".format(nr_queries))
    (_, tot_dim) = data.shape

    #Do you need to know which query you have? Which nr? Is that important? Seems like it isn't
    data_matrix = np.zeros(shape = (nr_queries, max_nr, dim_feature_size))
    data_matrix_r = np.zeros(shape = (nr_queries, max_nr))

    print('Padding documents')
    for i,query in enumerate(queries):
        if i % 1000:
            print("%d/%d" % (i, nr_queries), end='\r')
        doc_q = data[data[:,1]==query]
        doc_q = doc_q[:, tot_dim - dim_feature_size: tot_dim] #documents
        nr_doc = doc_q.shape[0]

        data_matrix[i,0:nr_doc,:] = doc_q

    print('Padding relevance labels')
    for i,query in enumerate(queries):
        if i % 1000:
                print("%d/%d" % (i, nr_queries), end='\r')
        q = data[data[:, 1] == query]
        nr_doc = q.shape[0]

        data_matrix_r[i, 0:nr_doc] = data[data[:, 1] == query,0]

    np.save("Web30KModified/small_max_zeros_data_{}".format(type), data_matrix)
    np.save("Web30KModified/small_max_zeros_rel_{}".format(type), data_matrix_r)

data_train = np.load("Web30KModified/web30k_small_max_doc_train.npy")
data_test = np.load("Web30KModified/web30k_small_max_doc_test.npy")
data_val = np.load("Web30KModified/web30k_small_max_doc_val.npy")

add_zeros(data_train, 136,'train')
add_zeros(data_test, 136,'test')
add_zeros(data_val, 136,'val')

def preprocessing_trec(train,dataset,type, i, year, dim):
    data_train = train.read()
    data_train = re.sub(r'\d+:', '', data_train)
    data_train = data_train.replace("qid:", "")
    data_train = data_train.replace("#docid = ", "")
    data_train = np.fromstring(data_train, dtype=float, sep=' ')
    data_train = np.reshape(data_train, (-1, dim+1))
    data_train = data_train[:, 0:dim]
    np.save("TD{}Modified/{}_data_{}_{}.npy".format(year, dataset,type, i), data_train)

def preprocessing_10k(file,dirs, type,dim):
    data_train = file.read()
    data_train = re.sub(r'\d+:', '', data_train)
    data_train = data_train.replace("qid:", "")
    data_train = np.fromstring(data_train, dtype=float, sep=' ')
    data_train = np.reshape(data_train, (-1, dim))
    np.save("Web30KModified/{}_data_{}.npy".format(dirs,type), data_train)

def preprocessing(file_train, file_test, file_val, dim_feature,file,fold_nr):
    data_train = np.load(file_train)
    data_test = np.load(file_test)
    data_val = np.load(file_val)

    queries_train = list(set(data_train[:, 1].astype(int)))
    queries_test = list(set(data_test[:, 1].astype(int)))
    queries_val = list(set(data_val[:, 1].astype(int)))

    data_all = np.concatenate((data_train, data_test, data_val))
    queries_all = queries_train + queries_test + queries_val
    max_nr = find_max_nr_doc(data_all, queries_all)

    (data_matrix_train, data_matrix_train_r) = add_zeros(data_train, queries_train, dim_feature, max_nr)
    (data_matrix_test, data_matrix_test_r) = add_zeros(data_test, queries_test, dim_feature, max_nr)
    (data_matrix_val, data_matrix_val_r) = add_zeros(data_val, queries_val, dim_feature, max_nr)

    np.save(file + "/" + fold_nr + "data_matrix_train", data_matrix_train)
    np.save(file + "/" + fold_nr + "data_matrix_train_r", data_matrix_train_r)
    np.save(file + "/" + fold_nr + "data_matrix_test", data_matrix_test)
    np.save(file + "/" + fold_nr + "data_matrix_test_r", data_matrix_test_r)
    np.save(file + "/" + fold_nr + "data_matrix_val", data_matrix_val)
    np.save(file + "/" + fold_nr + "data_matrix_val_r", data_matrix_val_r)
"""
file_test = "Web10kModified/1_data_TREC_test.npy"
file_train = "Web10kModified/1_data_TREC_train.npy"
file_val = "Web10kModified/1_data_TREC_val.npy"

preprocessing(file_train, file_test, file_val, 136, "Web10kModified", '1')

file_test = "Web10kModified/2_data_TREC_test.npy"
file_train = "Web10kModified/2_data_TREC_train.npy"
file_val = "Web10kModified/2_data_TREC_val.npy"

preprocessing(file_train, file_test, file_val, 136, "Web10kModified", '2')

file_test = "Web10kModified/3_data_TREC_test.npy"
file_train = "Web10kModified/3_data_TREC_train.npy"
file_val = "Web10kModified/3_data_TREC_val.npy"

preprocessing(file_train, file_test, file_val, 136, "Web10kModified", '3')

file_test = "Web10kModified/4_data_TREC_test.npy"
file_train = "Web10kModified/4_data_TREC_train.npy"
file_val = "Web10kModified/4_data_TREC_val.npy"

preprocessing(file_train, file_test, file_val, 136, "Web10kModified", '4')

file_test = "Web10kModified/5_data_TREC_test.npy"
file_train = "Web10kModified/5_data_TREC_train.npy"
file_val = "Web10kModified/5_data_TREC_val.npy"

preprocessing(file_train, file_test, file_val, 136, "Web10kModified", '5')
"""