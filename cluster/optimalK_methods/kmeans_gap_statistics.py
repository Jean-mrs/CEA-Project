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
    Wkbs = np.zeros(len(range(1, maxClusters)))
    Wks = np.zeros(len(range(1, maxClusters)))
    BWkbs = np.zeros(len(range(0, nrefs)))
    gp = pd.DataFrame({'clusterCount': [], 'Gap_sk': []})

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

            #New
            BWkbs[i] = np.log(km.inertia_)

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_

        #New
        Wks[gap_index] = np.log(km.inertia_)
        Wkbs[gap_index] = sum(BWkbs) / nrefs
        skd[gap_index] = np.sqrt(sum((BWkbs - Wkbs[gap_index]) ** 2) / nrefs)

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    #New
    sk = skd * np.sqrt(1 + 1 / nrefs)
    for k, gape in enumerate(gaps):
        if not k == len(gaps) - 1:
            gap_sk = gaps[k] - gaps[k + 1] - sk[k + 1]
        else:
            pass
            #gap_sk = gaps[k] - sk[k]
        gp = gp.append({'clusterCount': k, 'Gap_sk': gap_sk}, ignore_index=True)

    return (gaps.argmax() + 1, sk,
            resultsdf, gp)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

