import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def gap_statistics_kmeans(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    skd = np.zeros(len(range(1, maxClusters)))
    sk = np.zeros(len(range(0, maxClusters)))
    Wkbs = np.zeros(len(range(1, maxClusters)))
    Wks = np.zeros(len(range(1, maxClusters)))
    BWkbs = np.zeros(len(range(0, nrefs)))
    gaps_sks = np.zeros((len(range(1, maxClusters))))
    gp = pd.DataFrame({'clusterCount': [], 'Gap_sk': []})

    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):
        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            # refDisp = km.inertia_
            # refDisps[i] = refDisp

            #New
            BWkbs[i] = np.log(km.inertia_)

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_

        #New
        Wks[gap_index] = np.log(km.inertia_)
        Wkbs[gap_index] = sum(BWkbs) / nrefs

        # Calculate gap statistic
        gap = sum(BWkbs)/nrefs -Wks[gap_index]

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

        #New
        skd[gap_index] = np.sqrt((sum((BWkbs - Wkbs[gap_index]) ** 2)) / nrefs)
        sk[gap_index] = skd[gap_index] * np.sqrt(1 + 1 / nrefs)

    for k, gape in enumerate(range(1, maxClusters)):
        if not k == len(gaps) - 1:
            if gaps[k] >= gaps[k + 1] - sk[k + 1]:
                gaps_sks[k] = gaps[k] - gaps[k + 1] - sk[k + 1]
        else:
            gaps_sks[k] = -20

        # Assign new gap values calculated by simulation error and cluster count to plot bar graph
        gp = gp.append({'clusterCount': k + 1, 'Gap_sk': gaps_sks[k]}, ignore_index=True)

        # Assign best cluster numbers by gap values
    iter_points = [x[0] + 1 for x in sorted([y for y in enumerate(gaps)], key=lambda x: x[1], reverse=True)[:3]]

    # Assign best cluster numbers by gap values calculated with simulation error
    iter_points_sk = [x[0] + 1 for x in sorted([y for y in enumerate(gaps_sks)], key=lambda x: x[1], reverse=True)[:3]]

    a = list(filter(lambda g: g in iter_points, iter_points_sk))
    if len(a) is not 0:
        if not min(a) is 1:
            k = min(a)
        else:
            a.remove(1)
            k = min(a)
    else:
        k = min(iter_points_sk)

    return k, resultsdf, gp

