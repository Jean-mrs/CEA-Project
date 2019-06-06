import numpy as np
import pandas as pd
from cluster.fuzzycmeans.cmeans_algorithm import cmeans


def gap_statistics_fuzzy(data, nrefs=3, maxClusters=15):
    """
    Calculates Fuzzy C-Means optimal C using Gap Statistic from Sentelle, Hong, Georgiopoulos, Anagnostopoulos
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalC)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    Fpc = []
    for gap_index, c in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform cmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            cntr, u, u0, d, jm, p, fpc = cmeans(data=randomReference, c=c, m=2, error=0.005, maxiter=1000)

            refDisps = jm

        # Fit cluster to original data and create dispersion
        cntr, u, u0, d, jm, p, fpc = cmeans(data=data, c=c, m=2, error=0.005, maxiter=1000)
        Fpc.append(fpc)

        origDisp = jm

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(np.mean(origDisp))

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': c, 'gap': gap}, ignore_index=True)
    return [x[0]+1 for x in sorted([y for y in enumerate(gaps)], key=lambda x: x[1], reverse=True)[:3]], resultsdf, Fpc

    #return (gaps.argmax() + 1,
     #       resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

