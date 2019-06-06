import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from kneed import KneeLocator
from cluster.fuzzycmeans.cmeans_algorithm import cmeans
from cluster.optimalK_methods.fuzzy_gap_statistics import gap_statistics_fuzzy
import seaborn as sns
from sklearn.metrics import silhouette_score

# Data Setup
points = 1000
axis_range = 1000

for w in range(10):
    x = [random.randint(1, axis_range) for j in range(points)]
    y = [random.randint(1, axis_range) for i in range(points)]
    X = np.array(list(list(x) for x in zip(x, y)))
    X_df = pd.DataFrame(X)
    fig1 = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=10)
    #plt.savefig("/home/jean/CEA-ML/cluster/Output/Data_Users_" + str(axis_range) + '_Sim' + str(w))
    # #plt.savefig("/home/jean/Documentos/CEA-ML/cluster/Output/Data_Users_" + str(axis_range) + '_Sim' + str(w))

    xpts = X[:, 0]
    ypts = X[:, 1]
    alldata = np.vstack((xpts,  ypts))

    # Fuzzy Elbow Method
    wcss = []
    for i in range(1, 30):
        cntr9, u9, u09, d9, jm9, p9, fpc9 =cmeans(data=alldata, c=i, m=2, error=0.005, maxiter=1000, init=None)
        wcss.append(np.mean(jm9))
    plt.plot(range(1, 30), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')  # within cluster sum of squares
    plt.show()
    kn = KneeLocator(range(1, 30), wcss, curve='convex', direction='decreasing')
    n_clusters = kn.knee
    print(n_clusters)

    # # Silhouette Method
    # s = []
    # for n_clusters in range(1, 30):
    #     cntr, u, u0, d, jm, p, fpc = cmeans(data=alldata, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)
    #     labels = kmeans.labels_
    #     centroids = kmeans.cluster_centers_
    #     s.append(silhouette_score(X, labels, metric='euclidean'))
    # plt.plot(s)
    # plt.ylabel("Silouette")
    # plt.xlabel("k")
    # plt.title("Silouette for K-means cell's behaviour")
    # sns.despine()
    # plt.show()
    # clust = y.argmax()
    # print("number of k: " + str(clust))

    # Fuzzy Gap Statistics
    c, gapdfs, fpc = gap_statistics_fuzzy(X, nrefs=500, maxClusters=30)
    print(fpc)

    print('Final: ' + str((c[0] + n_clusters)/2))

    # Fuzzy C-Means Algorithm
    for i in range(len(c)):
        print('Optimal C is: ', c[i])
        fig2 = plt.figure()
        plt.plot(gapdfs.clusterCount, gapdfs.gap, linewidth=3)
        plt.scatter(gapdfs[gapdfs.clusterCount == c[i]].clusterCount, gapdfs[gapdfs.clusterCount == c[i]].gap, s=250, c='r')
        plt.grid(True)
        plt.xlabel('cluster Count')
        plt.ylabel('Gap Value')
        plt.title('cluster Number' + str(i) + '(Fuzzy C-Means)')
        #plt.savefig("/home/jean/Documentos/CEA-ML/cluster/Output/Gap_" + str(axis_range) + "_Best" + str(i) + '_Sim' + str(w))
        #plt.savefig("/home/jean/CEA-ML/cluster/Output/Gap_" + str(axis_range) + "_Best" + str(i) + '_Sim' + str(w))

        cntr, u, u0, d, jm, p, fpc = cmeans(data=alldata, c=c[i], m=2, error=0.005, maxiter=1000, init=None)
        print("Euclidian Distance Matrix: ")
        print(pd.DataFrame(d))
        print("\n")
        print("Final Matrix: ")
        print(pd.DataFrame(u))
        print("\n")

        # Plot Fuzzy
        #draw_model_2d(cntr, data=X, membership=np.transpose(u))

        # Save Output
        cntr_df = pd.DataFrame(cntr)
        #export_csv = cntr_df.to_csv(r'/home/jean/CEA-ML/cluster/Output/gw_' + str(axis_range) + "_Best" + str(i) + '_Sim' + str(w) + '.csv', header=True)
        #export_csv = cntr_df.to_csv(r'/home/jean/Documentos/CEA-ML/cluster/Output/gw_' + str(axis_range) + "_Best" + str(i) + '_Sim' + str(w) + '.csv', header=True)
    #export_csv2 = X_df.to_csv(r'/home/jean/CEA-ML/cluster/Output/User_' + str(axis_range) + '_Sim' + str(w) + '.csv', header=True)
    #export_csv2 = X_df.to_csv(r'/home/jean/Documentos/CEA-ML/cluster/Output/User_' + str(axis_range) + '_Sim' + str(w) + '.csv', header=True)