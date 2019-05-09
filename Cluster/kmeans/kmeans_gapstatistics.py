import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import seaborn as sns

from Cluster.optimalK_methods.gap_statistics import gap_statitics


x, y = make_blobs(750, n_features=2, centers=15)

plt.scatter(x[:, 0], x[:, 1])
plt.show()

print(x.shape)
print(y)
print("fim")

################ Elbow Method ###############
wcss = []
for i in range(1, 15):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # within cluster sum of squares
plt.show()

# Find Elbow/Knee value
kn = KneeLocator(range(1, 15), wcss, curve='convex', direction='decreasing')
n_clusters = kn.knee
print("Cluster Number: ", n_clusters)

df = pd.DataFrame(x, columns=['x', 'y'])

# Plot
kmea = KMeans(n_clusters)
kmea.fit(x)
df['label'] = kmea.labels_
colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(df.label.unique())))
#colors = plt.cm.Spectral(np.linspace(0, 1, len(df.label.unique())))
for color, label in zip(colors, df.label.unique()):
    tempdf = df[df.label == label]
    plt.scatter(tempdf.x, tempdf.y, c=color)
plt.scatter(kmea.cluster_centers_[:, 0], kmea.cluster_centers_[:, 1], c='r', s=50, alpha=0.7, )
plt.grid(True)
plt.show()

################# Gap Statitics Method ###############
k, gapdf = gap_statitics(x, nrefs=5, maxClusters=15)
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
km.fit(x)
print(km)
df['label'] = km.labels_
colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(df.label.unique())))
#colors = plt.cm.Spectral(np.linspace(0, 1, len(df.label.unique())))
for color, label in zip(colors, df.label.unique()):
    tempdf = df[df.label == label]
    plt.scatter(tempdf.x, tempdf.y, c=color)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='r', s=50, alpha=0.7, )
#plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1)
plt.grid(True)
plt.show()
#######################################################


########Silhouette Method########################
s = []
for n_clusters in range(2,30):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(x)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    s.append(silhouette_score(x, labels, metric='euclidean'))
plt.plot(s)
plt.ylabel("Silouette")
plt.xlabel("k")
plt.title("Silouette for K-means cell's behaviour")
sns.despine()
plt.show()
clust = y.argmax()
print("number of k: " + str(clust))
##################################################


"""
# plot the representation of the KMeans model
centers = km.cluster_centers_
radii = [cdist(x[df.label == i], km.cluster_centers_).max()
         for i, center in enumerate(centers)]
for c, r in zip(centers, radii):
    plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1)
plt.show()
"""



"""
num, gapdf = optimalK(km.cluster_centers_, maxClusters=5)
plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
plt.scatter(gapdf[gapdf.clusterCount == num].clusterCount, gapdf[gapdf.clusterCount == num].gap, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()


subKm = KMeans(n_clusters=num)
subKm.fit(km.cluster_centers_)

df = pd.DataFrame(km.cluster_centers_, columns=['x', 'y'])
df['label'] = subKm.labels_

colors = plt.cm.Spectral(np.linspace(0, 1, len(df.label.unique())))

for color, label in zip(colors, df.label.unique()):
    tempdf = df[df.label == label]
    plt.scatter(tempdf.x, tempdf.y, c=color, s=250)

plt.scatter(subKm.cluster_centers_[:, 0], subKm.cluster_centers_[:, 1], c='r', s=500, alpha=0.7, )
plt.grid(True)
plt.show()
"""