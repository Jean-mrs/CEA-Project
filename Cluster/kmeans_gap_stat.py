import numpy as np
import random

def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])
def bounding_box(X):
    xmin, xmax = min(X, key=lambda a: a[0])[0], max(X, key=lambda a: a[0])[0]
    ymin, ymax = min(X, key=lambda a: a[1])[1], max(X, key=lambda a: a[1])[1]
    return (xmin, xmax), (ymin, ymax)
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)
def gap_statistic(X):
    (xmin, xmax), (ymin, ymax) = bounding_box(X)
    # Dispersion for real distribution
    ks = range(1, 10)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    for indk, k in enumerate(ks):
        mu, clusters = find_centers(X, k)
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 10
        BWkbs = np.zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([np.random.uniform(xmin, xmax),
                           np.random.uniform(ymin, ymax)])
            Xb = np.array(Xb)
            mu, clusters = find_centers(Xb, k)
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs) / B
        sk[indk] = np.sqrt(sum((BWkbs - Wkbs[indk]) ** 2) / B)
    sk = sk * np.sqrt(1 + 1 / B)
    return (ks, Wks, Wkbs, sk)

def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        s = np.random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

X = init_board_gauss(200,3)
ks, logWks, logWkbs, sk = gap_statistic(X)
