import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import hashlib
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

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


# get optimal K for raw data
n, gapdf = optimalK(x, nrefs=3, maxClusters=15)
print ('Optimal K for raw data is: ', n)

# create KMeans given optimal n and fit
km = KMeans(n_clusters=n)
km.fit(x)

# assign points to clusters, hashing the cluster centers for the given label (used to join later)
df = pd.DataFrame(x, columns=['x','y'])
df['elementcluster'] = km.labels_
df['centroidHash'] = df.elementcluster.map(lambda label: hashlib.sha1(str(km.cluster_centers_[label]).encode('utf-8')).hexdigest())


# Find optimal clusters for cluster centers from above
n, gapdf = optimalK(km.cluster_centers_, nrefs=3, maxClusters=len(km.cluster_centers_))
print ('Optimal K for first clusters is: ', n)

# Create new KMeans subdfcluster df and assign cluster centers their own labels
km2 = KMeans(n_clusters=n)
km2.fit(km.cluster_centers_)

# Assign points to clusters, and hashing the original cluster centroids of original points to be used to join DFs
subdf = pd.DataFrame(km.cluster_centers_, columns=['cluster_x','cluster_y'])
subdf['groupcluster'] = km2.labels_
subdf['centroidHash'] = subdf.apply(lambda line: hashlib.sha1(str(np.array([line.cluster_x, line.cluster_y])).encode('utf-8')).hexdigest(), axis=1)

# Join dataframes
df = df.merge(subdf, how='left', on='centroidHash')

# Plot individual points colored to their cluster
colors = plt.cm.Spectral(np.linspace(0, 1, len(df.elementcluster.unique())))
for color, label in zip(colors, df.elementcluster.unique()):
    tempdf = df[df.elementcluster == label]
    plt.scatter(tempdf.x, tempdf.y, c=color)

# Plot the cluster centroids from the individual points
colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(df.groupcluster.unique())))
for color, label in zip(colors, df.groupcluster.unique()):
    tempdf = df[df.groupcluster == label]
    plt.scatter(tempdf.cluster_x, tempdf.cluster_y, c=color, s=500, alpha=0.8)

# Plot the centroids of those cluster centroids (clusters of the clusters)
plt.scatter(km2.cluster_centers_[:, 0], km2.cluster_centers_[:, 1], c='r', s=900, alpha=0.7, )

plt.show()

