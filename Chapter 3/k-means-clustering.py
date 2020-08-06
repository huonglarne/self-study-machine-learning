import numpy as np
import random
from sklearn import datasets
from scipy.spatial import distance
import matplotlib.pyplot as plt

# generate random data points
n_samples=1500
random_state=170
data, y = datasets.make_blobs(n_samples=n_samples, cluster_std=[2.5, 1.5, 0.5], random_state=random_state)
data = np.asarray(data)

k = 3

# choose k random centroids from the set
random_samples = random.sample(range(len(data)), k)
centroids = data[random_samples]

cluster_assignment = np.zeros((len(data)))
iter = 0

while True:
    old_cluster = cluster_assignment.copy()

    for i, point in enumerate(data):
        distances = [distance.euclidean(point, x) for x in centroids]
        cluster_assignment[i] = distances.index(min(distances))

    for i in range(k):
        same_cluster = (cluster_assignment==i)
        centroids[i] = [np.mean(data[same_cluster][0]), np.mean(data[same_cluster][1])]

    # see the difference between cluster assignment afer each iteration
    similarity = [(cluster_assignment[x] == old_cluster[x]) for x in range(len(data))]
    print(centroids, np.mean(similarity))
    iter += 1
    if np.mean(similarity)>0.99:
        break

print(iter)

# plot clusters. please change the color set according to the number k.
for i, color in zip(range(k), 'rgb'):
    same_cluster = (cluster_assignment==i)
    plt.scatter(data[same_cluster][:, 0], data[same_cluster][:, 1], c=color)
plt.scatter(centroids[:,0], centroids[:,1])
plt.show()

# please run this program multiple times because the result largely depends on the initial guess of centroids

