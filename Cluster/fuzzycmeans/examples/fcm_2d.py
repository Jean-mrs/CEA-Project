import numpy as np
import logging
from sklearn.datasets.samples_generator import make_blobs

x, y = make_blobs(2000, n_features=2, centers=4)

import sys
sys.path.append('..')


from Cluster.fuzzycmeans.fuzzy_clustering import FCM
from Cluster.fuzzycmeans.visualization import draw_model_2d


def example():
    X = np.array([[1, 1], [1, 2], [2, 2], [9, 10], [10, 10], [10, 9], [9, 9], [20,20]])
    #X = np.array(x)
    fcm = FCM(n_clusters=4)
    fcm.set_logger(tostdout=True, level=logging.DEBUG)
    #fcm.fit(X, [0, 0, 0, 1, 1, 1, 1, 2])
    fcm.fit(x, y)
    testing_data = x
    #testing_data = np.array([[0, 1.9], [5, 3], [4, 4], [8, 9], [9.5, 6.5], [5, 5], [15,15], [12,12], [14,14], [19,10]])
    predicted_membership = fcm.predict(testing_data)
    print("\n\ntesting data")
    print(testing_data)
    print("\npredicted membership")
    print(predicted_membership)
    print("\n\n")
    draw_model_2d(fcm, data=testing_data, membership=predicted_membership)

example()
