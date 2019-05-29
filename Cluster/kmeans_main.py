import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from Cluster.kmeans.kmeans import K_Means
from Cluster.optimalK_methods.kmeans_gap_statistics import gap_statistics_kmeans


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

# K-means Gap Statistics
k, gapdf = gap_statistics_kmeans(X, nrefs=5, maxClusters=30)
print('Optimal k is: ', k)
plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count (K-Means)')
plt.show()

# K-means Algorithm
km = K_Means(k)
km.fit(X)

# Plot K-means
colors = 10 * ["r", "g", "c", "b", "k"]
for centroid in km.centroids:
    plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s=130, marker="x")

for classification in km.classes:
    color = colors[classification]
    for features in km.classes[classification]:
        plt.scatter(features[0], features[1], color=color, s=30)
plt.show()

#
# df = pd.DataFrame(X, columns=['x', 'y'])
# df['label'] = km.
# colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(df.label.unique())))
# for color, label in zip(colors, df.label.unique()):
#     tempdf = df[df.label == label]
#     plt.scatter(tempdf.x, tempdf.y, c=color, s=10)
# plt.scatter(km.centroids[:, 0], km.centroids[:, 1], c='r', s=30, alpha=0.7, )
# plt.grid(True)
# plt.show()
