import numpy as np
import matplotlib.pyplot as plt
#import skfuzzy as fuzz
from Cluster.fuzzycmeans.cluster._cmeans import cmeans
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from Cluster.fuzzycmeans.visualization import draw_model_2d


#centers = [[4, 2], [1, 7], [5, 6]]

points = 10
n_clusters = 3

###Dataset Setup
df = pd.read_csv('/home/jean/Downloads/iris.data')
X = df.iloc[:, :2].values
#X = df.data[:, :2]
print(pd.DataFrame(df))
print(X)
x, y = make_blobs(points, n_features=2, centers=n_clusters)
print(x)
plt.scatter(x[:, 0], x[:, 1])
plt.show()

xpts = X[:, 0]
ypts = X[:, 1]
alldata = np.vstack((X[:, 0],  X[:, 1]))

###Data Arrangement
colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
# fig0, ax0 = plt.subplots()
# for label in range(n_clusters):
#     ax0.plot(xpts[y == label], ypts[y == label], '.', color=colors[label])
# ax0.set_title('Test data: {0} points x{1} clusters.'.format(points, n_clusters))
# plt.show()
# fig1, axes1 = plt.subplots()

###Fuzzy C-Means Algorithm
#cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data=alldata, c=3, m=2, error=0.005, maxiter=1000, init=None)
cntr, u, u0, d, jm, p, fpc = cmeans(data=alldata, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)
print(cntr)
print(pd.DataFrame(d))
print(pd.DataFrame(u))
print(jm)
print(p)
print(fpc)

###Cluster Membership
#cluster_membership = np.argmax(u, axis=0)
#cluster_membership = np.argwhere(u == np.amax(u, axis=0))
#print(cluster_membership.flatten().tolist())
#print(np.argmax(u, axis=0))
predict_memb = cmeans_predict(alldata, cntr_trained, m, error, maxiter, metric='euclidean', init=None,
                   seed=None)

###Plot Graph
draw_model_2d(fcm, data=alldata, membership=cluster_membership)


#
# for j in range(n_clusters):
#     axes1.plot(xpts[cluster_membership == j], ypts[cluster_membership == j], '.', color=colors[j])
# for pt in cntr:
#     axes1.plot(pt[0], pt[1], 'rs')
#     axes1.set_title('Centers = {0}; FPC = {1:.2f}'.format(3, fpc))
#plt.grid(True)
#plt.show()
