import numpy as np
import pandas as pd
from cluster.fuzzycmeans.cmeans_algorithm import cmeans


def gap_statistics_fuzzy(data, nnodes, nrefs=3, maxClusters=15):
    """
    Calculates Fuzzy C-Means optimal C using Gap Statistic from Sentelle, Hong, Georgiopoulos, Anagnostopoulos
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalC)
    """
    sdk = np.zeros(len(range(0, maxClusters)))
    sError = np.zeros(len(range(0, maxClusters)))
    BWkbs = np.zeros(len(range(0, nrefs)))
    Wks = np.zeros(len(range(0, maxClusters)))
    Wkbs = np.zeros(len(range(0, maxClusters)))
    gaps_sks = np.zeros((len(range(1, maxClusters))))
    gp = pd.DataFrame({'clusterCount': [], 'Gap_sk': []})

    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})

    for gap_index, c in enumerate(range(1, maxClusters)):

        # For n references, generate random sample and perform cmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            cntr1, u1, u01, d1, jm, p1, fpc1 = cmeans(data=randomReference, c=c, m=2, error=0.005, maxiter=1000,
                                                       nnodes=nnodes)
            cntr, u, u0, d, jm, p, fpc = cmeans(data=randomReference.T, c=c, m=2, error=0.005, maxiter=1000, nnodes=nnodes,
                                                bstation=cntr1, testsensi=True)


            # Holder for reference dispersion results
            BWkbs[i] = np.log(jm[len(jm)-1])
            #BWkbs[i] = np.log(np.mean(jm))

        # Fit cluster to original data and create dispersion
        cntr1, u1, u01, d1, jm1, p1, fpc1 = cmeans(data=data, c=c, m=2, error=0.005, maxiter=1000, nnodes=nnodes)
        cntr, u, u0, d, jm, p, fpc = cmeans(data=data.T, c=c, m=2, error=0.005, maxiter=1000, nnodes=nnodes,
                                            bstation=cntr1, testsensi=True)

        # Holder for original dispersion results
        Wks[gap_index] = np.log(jm[len(jm)-1])
        #Wks[gap_index] = np.log(np.mean(jm))

        # Calculate gap statistic
        gap = sum(BWkbs)/nrefs - Wks[gap_index]

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        # Assign gap values and cluster count to plot curve graph
        resultsdf = resultsdf.append({'clusterCount': c, 'gap': gap}, ignore_index=True)

        # Compute Standard Deviation
        Wkbs[gap_index] = sum(BWkbs) / nrefs
        sdk[gap_index] = np.sqrt((sum((BWkbs - Wkbs[gap_index])**2))/nrefs)

        # Compute the Simulation Error
        sError[gap_index] = sdk[gap_index] * np.sqrt(1 + 1 / nrefs)

    for k, gape in enumerate(range(1, maxClusters)):
        if not k == len(gaps) - 1:
            if gaps[k] >= gaps[k + 1] - sError[k + 1]:
                gaps_sks[k] = gaps[k] - gaps[k + 1] - sError[k + 1]
        else:
            gaps_sks[k] = -20

        # Assign new gap values calculated by simulation error and cluster count to plot bar graph
        gp = gp.append({'clusterCount': k+1, 'Gap_sk': gaps_sks[k]}, ignore_index=True)

    # Assign best cluster numbers by gap values
    iter_points = [x[0]+1 for x in sorted([y for y in enumerate(gaps)], key=lambda x: x[1], reverse=True)[:3]]

    # Assign best cluster numbers by gap values calculated with simulation error
    iter_points_sk = [x[0]+1 for x in sorted([y for y in enumerate(gaps_sks)], key=lambda x: x[1], reverse=True)[:3]]

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

