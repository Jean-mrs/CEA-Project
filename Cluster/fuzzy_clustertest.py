import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pprint
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
#centers = [[4, 2], [1, 7], [5, 6]]


centers = 3
x, y = make_blobs(5, n_features=2, centers=centers)
plt.scatter(x[:, 0], x[:, 1])
plt.show()

xpts = x[:, 0]
ypts = x[:, 1]

fig0, ax0 = plt.subplots()
for label in range(centers):
    ax0.plot(xpts[y == label], ypts[y == label], '.', color=colors[label])
ax0.set_title('Test data: 200 points x{0} clusters.'.format(centers))
plt.show()
fig1, axes1 = plt.subplots()
alldata = np.vstack((x[:, 0],  x[:, 1]))
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data=alldata, c=3, m=2, error=0.005, maxiter=1000, init=None)
print(cntr)
print(pd.DataFrame(u))
print(pd.DataFrame(d))
print(jm)
print(p)
print(fpc)

#cluster_membership = np.argmax(u, axis=0)
cluster_membership = np.argwhere(u == np.amax(u, axis=0))
print(cluster_membership.flatten().tolist())
print(np.argmax(u, axis=0))

for j in range(centers):
    axes1.plot(xpts[cluster_membership == j], ypts[cluster_membership == j], '.', color=colors[j])
for pt in cntr:
    axes1.plot(pt[0], pt[1], 'rs')
    axes1.set_title('Centers = {0}; FPC = {1:.2f}'.format(3, fpc))
plt.grid(True)
plt.show()
