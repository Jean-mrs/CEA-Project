import numpy as np
import matplotlib.pyplot as plt
#import skfuzzy as fuzz
import pandas as pd
import random
from sklearn.cluster import KMeans
from Cluster.fuzzycmeans.cluster._cmeans import cmeans
from Cluster.fuzzycmeans.cluster._cmeans import cmeans_predict
from Cluster.optimalK_methods.kmeans_gap_statistics import gap_statistics_kmeans
from Cluster.optimalK_methods.fuzzy_gap_statistics import gap_statistics_fuzzy
from Cluster.fuzzycmeans.cluster.visualization_graph import draw_model_2d



points = 20
#n_clusters = 10

###Dataset Setup
x = [random.randint(1, 10000) for j in range(points)]
y = [random.randint(1, 10000) for i in range(points)]
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


################# Gap Statitics Method ###############
k, gapdf = gap_statistics_kmeans(X, nrefs=5, maxClusters=15)
print('Optimal k is: ', k)
plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()

# Plot
km = KMeans(k)
km.fit(X)
print(km)
df = pd.DataFrame(X, columns=['x', 'y'])
df['label'] = km.labels_
colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(df.label.unique())))
for color, label in zip(colors, df.label.unique()):
    tempdf = df[df.label == label]
    plt.scatter(tempdf.x, tempdf.y, c=color)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='r', s=50, alpha=0.7, )
plt.grid(True)
plt.show()

###Fuzzy C-Means Algorithm
###Fuzzy Gap Statitics Method
c, gapdfs = gap_statistics_fuzzy(X, nrefs=5, maxClusters=15)
print('Optimal C is: ', c)
plt.plot(gapdfs.clusterCount, gapdfs.gap, linewidth=3)
plt.scatter(gapdf[gapdfs.clusterCount == c].clusterCount, gapdf[gapdfs.clusterCount == c].gap, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()

###Fuzzy C-Means Algorithm
cntr, u, u0, d, jm, p, fpc = cmeans(data=alldata, c=c, m=2, error=0.005, maxiter=1000, init=None)
#print("Centers Coordinates:")
#print(cntr)
print("\n")
#print("Original Matrix: ")
#print(pd.DataFrame(u0))
print("\n")
print("Final Matrix: ")
print(pd.DataFrame(u))
print("\n")
#print("Object Function: ")
#print(jm)
print("\n")
print("Euclidian Distance Matrix: ")
print(pd.DataFrame(d))
print("\n")
#print(p)
#print(fpc)

matrix, matrix0, d, jm1, p2, fpc2 = cmeans_predict(alldata, cntr_trained=cntr, m=2, error=0.005, maxiter=1000, init=u)
print("predict:")
print(pd.DataFrame(np.transpose(matrix)))
print(pd.DataFrame(d))
print(type(d))
###Plot Graph
#draw_model_2d(cntr, data=X, membership=np.transpose(matrix))

