import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from Cluster.fuzzycmeans.cluster._cmeans import cmeans
from Cluster.optimalK_methods.fuzzy_gap_statistics import gap_statistics_fuzzy
from Cluster.fuzzycmeans.cluster.visualization_graph import draw_model_2d


# Data Setup
points = 1000
axis_range = 10000

for w in range(5):
    x = [random.randint(1, axis_range) for j in range(points)]
    y = [random.randint(1, axis_range) for i in range(points)]
    X = np.array(list(list(x) for x in zip(x, y)))
    X_df = pd.DataFrame(X)
    fig1 = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.savefig("/home/jean/Documentos/CEA-ML/Cluster/Output/Data_Users_" + str(axis_range) + '_Sim' + str(w))

    xpts = X[:, 0]
    ypts = X[:, 1]
    alldata = np.vstack((xpts,  ypts))

    # Fuzzy Gap Statistics
    c, gapdfs = gap_statistics_fuzzy(X, nrefs=5, maxClusters=30)

    # Fuzzy C-Means Algorithm
    for i in range(len(c)):
        print('Optimal C is: ', c[i])
        fig2 = plt.figure()
        plt.plot(gapdfs.clusterCount, gapdfs.gap, linewidth=3)
        plt.scatter(gapdfs[gapdfs.clusterCount == c[i]].clusterCount, gapdfs[gapdfs.clusterCount == c[i]].gap, s=250, c='r')
        plt.grid(True)
        plt.xlabel('Cluster Count')
        plt.ylabel('Gap Value')
        plt.title('Cluster Number' + str(i) + '(Fuzzy C-Means)')
        plt.savefig("/home/jean/Documentos/CEA-ML/Cluster/Output/Gap_" + str(axis_range) + "_Best" + str(i) + '_Sim' + str(w))

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
        export_csv = cntr_df.to_csv(r'/home/jean/Documentos/CEA-ML/Cluster/Output/gw_' + str(axis_range) + "_Best" + str(i) + '_Sim' + str(w) + '.csv', header=True)
    export_csv2 = X_df.to_csv(r'/home/jean/Documentos/CEA-ML/Cluster/Output/User_' + str(axis_range) + '_Sim' + str(w) + '.csv', header=True)