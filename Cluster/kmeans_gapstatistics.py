import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from kneed import KneeLocator

x, y = make_blobs(750, n_features=2, centers=15)

plt.scatter(x[:, 0], x[:, 1])
plt.show()


def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    return (gaps.argmax() + 1,
            resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


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
###############################################


k, gapdf = optimalK(x, nrefs=5, maxClusters=15)
print('Optimal k is: ', k)
plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()

km = KMeans(k)
km.fit(x)

print(km)
df = pd.DataFrame(x, columns=['x', 'y'])
df['label'] = km.labels_

colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(df.label.unique())))
#colors = plt.cm.Spectral(np.linspace(0, 1, len(df.label.unique())))

print(colors)
for color, label in zip(colors, df.label.unique()):
    tempdf = df[df.label == label]
    plt.scatter(tempdf.x, tempdf.y, c=color)

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='r', s=50, alpha=0.7, )
plt.grid(True)
plt.show()


##############Elbow Kmean Plot ##################
kmea = KMeans(n_clusters)
kmea.fit(x)
df['label'] = kmea.labels_
colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(df.label.unique())))
#colors = plt.cm.Spectral(np.linspace(0, 1, len(df.label.unique())))
print(colors)
for color, label in zip(colors, df.label.unique()):
    tempdf = df[df.label == label]
    plt.scatter(tempdf.x, tempdf.y, c=color)
plt.scatter(kmea.cluster_centers_[:, 0], kmea.cluster_centers_[:, 1], c='r', s=50, alpha=0.7, )
plt.grid(True)
plt.show()
##################################################


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