import numpy as np
import matplotlib.pyplot as plt
#import skfuzzy as fuzz
from Cluster.fuzzycmeans.cmeans import cmeans
from Cluster.fuzzycmeans.cmeans import cmeans_predict
import pandas as pd
from Cluster.fuzzycmeans.visualization_graph import draw_model_2d
import random


points = 1000
n_clusters = 10

###Dataset Setup
#x, y = make_blobs(points, n_features=2, centers=n_clusters)
x = [random.randint(1,10000000) for j in range(points)]
y = [random.randint(1,10000000) for i in range(points)]
X = np.array(list(list(x) for x in zip(x, y)))
print(type(X))
print(X)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

xpts = X[:, 0]
ypts = X[:, 1]
alldata = np.vstack((X[:, 0],  X[:, 1]))
print("Alldata:")
print(alldata)

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
print("Centers Coordinates:")
print(cntr)
print("\n")
print("Original Matrix: ")
print(pd.DataFrame(d))
print("\n")
print("Final Matrix: ")
print(pd.DataFrame(u))
print("\n")
print(jm)
print(p)
print(fpc)

###Cluster Membership
#cluster_membership = np.argmax(u, axis=0)
#cluster_membership = np.argwhere(u == np.amax(u, axis=0))
#print(cluster_membership.flatten().tolist())
#print(np.argmax(u, axis=0))
matrix, matrix0, d, jm, p, fpc = cmeans_predict(alldata, cntr_trained=cntr, m=2, error=0.005, maxiter=1000, init=u)
print("predict:")
print(pd.DataFrame(np.transpose(matrix)))
###Plot Graph
draw_model_2d(cntr, data=X, membership=np.transpose(matrix))


#
# for j in range(n_clusters):
#     axes1.plot(xpts[cluster_membership == j], ypts[cluster_membership == j], '.', color=colors[j])
# for pt in cntr:
#     axes1.plot(pt[0], pt[1], 'rs')
#     axes1.set_title('Centers = {0}; FPC = {1:.2f}'.format(3, fpc))
#plt.grid(True)
#plt.show()
