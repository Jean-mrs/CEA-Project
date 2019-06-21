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
    sdk = np.zeros(len(range(0, maxClusters)))
    sError = np.zeros(len(range(0, maxClusters)))
    BWkbs = np.zeros(len(range(0, nrefs)))
    Wks = np.zeros(len(range(0, maxClusters)))
    Wkbs = np.zeros(len(range(0, maxClusters)))
    gaps_sks = np.zeros((len(range(1, maxClusters))))
    gp = pd.DataFrame({'clusterCount': [], 'Gap_sk': []})

    gaps = np.zeros((len(range(1, maxClusters)),))
    #best = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})

    for gap_index, c in enumerate(range(1, maxClusters)):

        # For n references, generate random sample and perform cmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            cntr, u, u0, d, jm, p, fpc = cmeans(data=randomReference, c=c, m=2, error=0.005, maxiter=1000)

            BWkbs[i] = np.log(jm[len(jm)-1])
            #BWkbs[i] = np.log(np.mean(jm))

        # Fit cluster to original data and create dispersion
        cntr, u, u0, d, jm, p, fpc = cmeans(data=data, c=c, m=2, error=0.005, maxiter=1000)

        Wks[gap_index] = np.log(jm[len(jm)-1])
        #Wks[gap_index] = np.log(np.mean(jm))

        # Calculate gap statistic
        gap = sum(BWkbs)/nrefs - Wks[gap_index]

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': c, 'gap': gap}, ignore_index=True)

        # Compute Standard Deviation
        Wkbs[gap_index] = sum(BWkbs) / nrefs
        sdk[gap_index] = np.sqrt((sum((BWkbs - Wkbs[gap_index])**2))/nrefs)

        # Compute the Simulation Error
        sError[gap_index] = sdk[gap_index] * np.sqrt(1 + 1 / nrefs)

    # for k, gape in enumerate(gaps):
    #     if not k == len(gaps) - 1:
    #         if gaps[k] >= gaps[k + 1] - sError[k + 1]:
    #             print(str(gaps[k]) + ' >= ' + str(gaps[k + 1]) + ' - ' + str(sError[k + 1]))
    #             best[k] = gaps[k]
    #     else:
    #         best[k] = -20
    #     gp = gp.append({'clusterCount': k, 'Gap_sk': best[k]}, ignore_index=True)

    for k, gape in enumerate(range(1, maxClusters)):
        if not k == len(gaps) - 1:
        #if not k == maxClusters - 1 and not k == maxClusters - 2:
            if gaps[k] >= gaps[k + 1] - sError[k + 1]:
                gaps_sks[k] = gaps[k] - gaps[k + 1] - sError[k + 1]
        else:
            gaps_sks[k] = -20
            #else:
         #   gaps_sks[k] = -1000
        gp = gp.append({'clusterCount': k, 'Gap_sk': gaps_sks[k]}, ignore_index=True)

    iter_points = [x[0]+1 for x in sorted([y for y in enumerate(gaps)], key=lambda x: x[1], reverse=True)[:3]]

    iter_points_sk = [x[0]+1 for x in sorted([y for y in enumerate(gaps_sks)], key=lambda x: x[1], reverse=True)[:3]]

    # if gaps_sks.argmax() <= 1:
    #     ga = list(gaps_sks)
    #     ga.pop(0)
    #     gaps_sks = np.array(ga)

    return iter_points, iter_points_sk, resultsdf, gp

