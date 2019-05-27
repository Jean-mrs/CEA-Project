import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.cluster import KMeans
from Cluster.fuzzycmeans.cluster._cmeans import cmeans
from Cluster.optimalK_methods.kmeans_gap_statistics import gap_statistics_kmeans
from Cluster.optimalK_methods.fuzzy_gap_statistics import gap_statistics_fuzzy
from Cluster.fuzzycmeans.cluster.visualization_graph import draw_model_2d


points = 1000

# Data Setup
x = [random.randint(1, 10000) for j in range(points)]
y = [random.randint(1, 10000) for i in range(points)]
X = np.array(list(list(x) for x in zip(x, y)))
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.show()

xpts = X[:, 0]
ypts = X[:, 1]
alldata = np.vstack((xpts,  ypts))

# # K-means Gap Statistics
# k, gapdf = gap_statistics_kmeans(X, nrefs=5, maxClusters=30)
# print('Optimal k is: ', k)
# plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
# plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
# plt.grid(True)
# plt.xlabel('Cluster Count')
# plt.ylabel('Gap Value')
# plt.title('Gap Values by Cluster Count (K-Means)')
# plt.show()
#
# # K-means Algorithm
# km = KMeans(k)
# km.fit(X)
# print(km)
#
# # Plot K-means
# df = pd.DataFrame(X, columns=['x', 'y'])
# df['label'] = km.labels_
# colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(df.label.unique())))
# for color, label in zip(colors, df.label.unique()):
#     tempdf = df[df.label == label]
#     plt.scatter(tempdf.x, tempdf.y, c=color, s=10)
# plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='r', s=30, alpha=0.7, )
# plt.grid(True)
# plt.show()

# # Fuzzy Gap Statistics
# c, gapdfs = gap_statistics_fuzzy(X, nrefs=5, maxClusters=30)
# print('Optimal C is: ', c)
# plt.plot(gapdfs.clusterCount, gapdfs.gap, linewidth=3)
# plt.scatter(gapdfs[gapdfs.clusterCount == c[0]].clusterCount, gapdfs[gapdfs.clusterCount == c[0]].gap, s=250, c='r')
# plt.grid(True)
# plt.xlabel('Cluster Count')
# plt.ylabel('Gap Value')
# plt.title('Gap Values by Cluster Count (Fuzzy C-Means)')
# plt.show()

# Fuzzy C-Means Algorithm

cntr, u, u0, d, jm, p, fpc = cmeans(data=alldata, c=10, m=2, error=0.005, maxiter=1000, init=None)
print("Euclidian Distance Matrix: ")
print(pd.DataFrame(d))
print("\n")
print("Final Matrix: ")
print(pd.DataFrame(u))
print("\n")

# Plot Fuzzy
#draw_model_2d(cntr, data=X, membership=np.transpose(u))

# Save Output
X_df = pd.DataFrame(X)
cntr_df = pd.DataFrame(cntr)
export_csv = cntr_df.to_csv(r'/home/jean/Documentos/CEA-ML/Cluster/Output/gateways'  + '.csv', header=True)
export_csv2 = X_df.to_csv(r'/home/jean/Documentos/CEA-ML/Cluster/Output/devices' + '.csv', header=True)
