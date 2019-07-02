import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import random
from cluster.fuzzycmeans.cmeans_algorithm import cmeans, points_limit
from cluster.optimalK_methods.fuzzy_gap_statistics import gap_statistics_fuzzy


# Data Setup
points = 1000
axis_range = 10000
#X, y = make_blobs(1000, n_features=2, centers=15)

for w in range(33):
    x = [random.randint(1, axis_range) for j in range(points)]
    y = [random.randint(1, axis_range) for i in range(points)]
    X = np.array(list(list(x) for x in zip(x, y)))
    X_df = pd.DataFrame(X)
    fig1 = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=4)
    plt.savefig("/home/jean/public_html/Sim_1000x33_Limit80/Data_Users_" + str(points) + '_Sim' + str(w))
    #plt.savefig("/home/jean/Documentos/CEA-ML/cluster/Output/Data_Users_" + str(axis_range) + '_Sim' + str(w))

    xpts = X[:, 0]
    ypts = X[:, 1]
    alldata = np.vstack((xpts,  ypts))

    # Fuzzy Gap Statistics
    k, gapdfs1, gapsk6 = gap_statistics_fuzzy(X, nrefs=500, maxClusters=30)
    print(k)
    print(gapdfs1)
    print(gapsk6)
    c = points_limit(data=alldata, k=k, maxPoints=80)
    print('Old C: ', k)

    # Plot Gap Statistics
    print('Optimal C is: ', c)
    fig2 = plt.figure()
    plt.plot(gapdfs1.clusterCount, gapdfs1.gap, linewidth=3)
    plt.scatter(gapdfs1[gapdfs1.clusterCount == c].clusterCount, gapdfs1[gapdfs1.clusterCount == c].gap, s=250, c='r')
    plt.grid(True)
    plt.xlabel('cluster Count')
    plt.ylabel('Gap Value')
    plt.title('cluster Number' + '(Fuzzy C-Means)')
    #plt.savefig("/home/jean/Documentos/CEA-ML/cluster/Output/Gap_" + str(axis_range) + "_Best" + '_Sim' + str(w))
    plt.savefig("/home/jean/public_html/Sim_1000x33_Limit80/Gap_" + str(points) + "_Best" + '_Sim' + str(w))

    # Plot Gap Error bars
    fig3 = plt.figure()
    plt.bar(gapsk6.clusterCount, gapsk6.Gap_sk)
    plt.xlabel('cluster Count')
    plt.ylabel('gap[k] - gap[k+1] - sk[k+1]')
    #plt.savefig("/home/jean/Documentos/CEA-ML/cluster/Output/Gap_Final_Values:" + str(axis_range) + "_Best" + '_Sim' + str(w))
    plt.savefig("/home/jean/public_html/Sim_1000x33_Limit80/Gap_Final_Values:" + str(points) + "_Best" + '_Sim' + str(w))

    # Fuzzy C-Means Algorithm
    cntr, u, u0, d, jm, p, fpc = cmeans(data=alldata, c=c, m=2, error=0.005, maxiter=1000, init=None)
    #print(cntr)
    #print("Euclidian Distance Matrix: ")
    #print(pd.DataFrame(d))
    #print("\n")
    print("Final Matrix: ")
    print(pd.DataFrame(u))
    print("\n")

    # Plot Fuzzy
    #draw_model_2d(cntr, data=X, membership=np.transpose(u))

    # Save Output
    cntr_df = pd.DataFrame(cntr)
    export_csv = cntr_df.to_csv(r'/home/jean/public_html/Sim_1000x33_Limit80/gw_' + str(points) + "_Best" + '_Sim' + str(w) + '.csv', header=True)
    #export_csv = cntr_df.to_csv(r'/home/jean/Documentos/CEA-ML/cluster/Output/gw_' + str(axis_range) + "_Best" + '_Sim' + str(w) + '.csv', header=True)
    export_csv2 = X_df.to_csv(r'/home/jean/public_html/Sim_1000x33_Limit80/User_' + str(points) + '_Sim' + str(w) + '.csv', header=True)
    #export_csv2 = X_df.to_csv(r'/home/jean/Documentos/CEA-ML/cluster/Output/User_' + str(axis_range) + '_Sim' + str(w) + '.csv', header=True)