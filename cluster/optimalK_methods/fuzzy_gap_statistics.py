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
    skd = np.zeros(len(range(1, maxClusters)))
    Wkbs = np.zeros(len(range(1, maxClusters)))
    Wks = np.zeros(len(range(1, maxClusters)))
    BWkbs = np.zeros(len(range(0, nrefs)))
    gaps_sks = np.zeros((len(range(1, maxClusters))))
    gp = pd.DataFrame({'clusterCount': [], 'Gap_sk': []})

    gaps = np.zeros((len(range(1, maxClusters)),))
    best = np.zeros((len(range(1, maxClusters)),))
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

            refDisps[i] = np.mean(jm)

            BWkbs[i] = np.mean([np.log(x) for x in jm])

        # Fit cluster to original data and create dispersion
        cntr, u, u0, d, jm, p, fpc = cmeans(data=data, c=c, m=2, error=0.005, maxiter=1000)
        Fpc.append(fpc)

        origDisp = jm

        Wks[gap_index] = np.mean([np.log(x) for x in jm])
        Wkbs[gap_index] = sum(BWkbs) / nrefs
        skd[gap_index] = np.sqrt(sum((BWkbs - Wkbs[gap_index]) ** 2) / nrefs)

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(np.mean(origDisp))

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': c, 'gap': gap}, ignore_index=True)

    sk = skd * np.sqrt(1 + 1 / nrefs)

    for k, gape in enumerate(gaps):
        if not k == len(gaps) - 1:
            if gaps[k] >= gaps[k + 1] - sk[k + 1]:
                print(str(gaps[k]) + ' >= ' + str(gaps[k + 1]) + ' - ' + str(sk[k + 1]))
                best[k] = gaps[k]
        else:
            best[k] = -20

    for k, gape in enumerate(range(1, maxClusters)):
        if not k == maxClusters - 1 and not k == maxClusters - 2:
            if gaps[k] >= gaps[k + 1] - sk[k + 1]:
                gap_sk = gaps[k] - gaps[k + 1] - sk[k + 1]
                gaps_sks[k] = gap_sk
        else:
            gaps_sks[k] = -1000
        gp = gp.append({'clusterCount': k, 'Gap_sk': gap_sk}, ignore_index=True)

    iter_points = [x[0]+1 for x in sorted([y for y in enumerate(gaps)], key=lambda x: x[1], reverse=True)[:3]]

    #iter_points_sk = [x[0]+1 for x in sorted([y for y in enumerate(gaps)], key=lambda x: x[0], reverse=True)]
    if gaps_sks.argmax() <= 1:
        ga = list(gaps_sks)
        ga.pop(0)
        gaps_sks = np.array(ga)

    return iter_points, gaps_sks.argmax() + 1, resultsdf, Fpc, gp, best

    #return (gaps.argmax() + 1,
     #       resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

