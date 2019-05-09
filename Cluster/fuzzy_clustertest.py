import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs

x, y = make_blobs(750, n_features=2, centers=12)
plt.scatter(x[:, 0], x[:, 1])
plt.show()

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data=x, c=12, m=2, error=0.005, maxiter=15, init=None)
print(fpc)



plt.scatter(cntr[:, 0], cntr[:, 1], c='r', s=50, alpha=0.7, )
plt.grid(True)
plt.show()