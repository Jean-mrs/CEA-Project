from pprint import pprint

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import random
import sys
from cluster.fuzzycmeans.cmeans_algorithm import cmeans
from cluster.optimalK_methods.fuzzy_gap_statistics import gap_statistics_fuzzy

# Data Setup
points = 7
axis_range = 10000
#X, y = make_blobs(1000, n_features=2, centers=15)

for w in range(1):
    # x = [random.randint(1, axis_range) for j in range(points)]
    # y = [random.randint(1, axis_range) for i in range(points)]
    # X = np.array(list(list(x) for x in zip(x, y)))
    x = [1, 100, 120, 110, 840, 8000, 400]
    y = [1, 100, 120, 110, 840, 8000, 400]
    X = np.array(list(list(a) for a in zip(x, y)))
    fig1 = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=4)
    #plt.savefig("/home/jean/public_html/Sim_2000x33_Limit70/Data_Users_" + str(points) + '_Sim' + str(w))
    #plt.savefig("/home/jean/Documents/CEA-Project/Output/Data_Users_" + str(axis_range) + '_Sim' + str(w))

    xpts = X[:, 0]
    ypts = X[:, 1]
    alldata = np.vstack((xpts,  ypts))

    # Fuzzy Gap Statistics
    k, gapdfs1, gapsk6 = gap_statistics_fuzzy(X, nrefs=1, maxClusters=30, nnodes=points)
    print('New C: ', k)

    # Plot Gap Statistics
    print('Optimal C is: ', k)
    fig2 = plt.figure()
    plt.plot(gapdfs1.clusterCount, gapdfs1.gap, linewidth=3)
    plt.scatter(gapdfs1[gapdfs1.clusterCount == k].clusterCount, gapdfs1[gapdfs1.clusterCount == k].gap, s=250, c='r')
    plt.grid(True)
    plt.xlabel('cluster Count')
    plt.ylabel('Gap Value')
    plt.title('cluster Number' + '(Fuzzy C-Means)')
    plt.savefig("/home/jean/Documents/CEA-Project/cluster/Output/Gap_" + str(axis_range) + "_Best" + '_Sim' + str(w))
    #plt.savefig("/home/jean/public_html/Sim_2000x33_Limit70/Gap_" + str(points) + "_Best" + '_Sim' + str(w))

    # Plot Gap Error bars
    fig3 = plt.figure()
    plt.bar(gapsk6.clusterCount, gapsk6.Gap_sk)
    plt.xlabel('cluster Count')
    plt.ylabel('gap[k] - gap[k+1] - sk[k+1]')
    plt.savefig("/home/jean/Documents/CEA-Project/cluster/Output/Gap_Final_Values:" + str(axis_range) + "_Best" + '_Sim' + str(w))
    #plt.savefig("/home/jean/public_html/Sim_2000x33_Limit70/Gap_Final_Values:" + str(points) + "_Best" + '_Sim' + str(w))

    # Fuzzy C-Means Algorithm
    cntr, u, u0, d, jm, p, fpc = cmeans(data=alldata, c=25, m=2, error=0.005, maxiter=2, init=None, nnodes=points)

    cntr1, u1, u01, d1, jm1, p1, fpc1 = cmeans(data=alldata, c=25, m=2, error=0.005, maxiter=2, init=None, nnodes=points,testsensi=True, bstation=cntr)
    print(pd.DataFrame(d1))
    #print("Euclidian Distance Matrix: ")
    #pprint(cntr)
    #print("\n")
    print("Final Matrix: ")
    print("\n")

    # Plot Fuzzy
    #draw_model_2d(cntr, data=X, membership=np.transpose(u))

    # Save Output
    cntr_df = pd.DataFrame(cntr1)
    #export_csv = cntr_df.to_csv(r'/home/jean/public_html/Sim_2000x33_Limit70/gw_' + str(points) + "_Best" + '_Sim' + str(w) + '.csv', header=True)
    #export_csv = cntr_df.to_csv(r'/home/jean/Documentos/CEA-ML/cluster/Output/gw_' + str(axis_range) + "_Best" + '_Sim' + str(w) + '.csv', header=True)
    #export_csv2 = X_df.to_csv(r'/home/jean/public_html/Sim_2000x33_Limit70/User_' + str(points) + '_Sim' + str(w) + '.csv', header=True)
    #export_csv2 = X_df.to_csv(r'/home/jean/Documentos/CEA-ML/cluster/Output/User_' + str(axis_range) + '_Sim' + str(w) + '.csv', header=True)