import pyltr
import numpy as np
from sklearn import preprocessing

test = np.load("Web30KModified/web30k_small_max_doc_test.npy")
train = np.load("Web30KModified/web30k_small_max_doc_train.npy")
val = np.load("Web30KModified/web30k_small_max_doc_val.npy")

TX = train[:,2:138]
Ty = train[:,0]
Tqids = train[:,1]
scaler = preprocessing.StandardScaler().fit(TX)
TX = scaler.transform(TX)

VX = val[:,2:138]
Vy = val[:,0]
Vqids = val[:,1]
VX = scaler.transform(VX)

EX = test[:,2:138]
Ey = test[:,0]
Eqids = test[:,1]
EX = scaler.transform(EX)

metric = pyltr.metrics.AP(k=100000)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.02,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
)

model.fit(TX, Ty, Tqids)

Epred = model.predict(EX)
print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
print('Our model:', metric.calc_mean(Eqids, Ey, Epred))

