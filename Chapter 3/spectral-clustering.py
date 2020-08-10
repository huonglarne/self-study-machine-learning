import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from scipy.spatial import distance

class spectral_clustering:
    def __init__(self, data):
        self.data = data
        self.n_samples = len(data)
        self.graph = np.zeros((n_samples, n_samples))
        """
        graph_type: epsilon neighborhood, k-nearest neighbors, fully connected
        """

    def build_graph(self, neighborhood=None, radius=None, k=None):
        if neighborhood=="in_radius":
            for i in range(self.n_samples):
                for j in range(i):
                    d = distance.euclidean(self.data[i], self.data[j])
                    if d <= radius:
                        self.graph[i,j] = -1
                        self.graph[j,i] = -1

        if neighborhood=="k_nearest":
            for i in range(self.n_samples):
                distances = [[distance.euclidean(self.data[j], self.data[i]), j] for j in range(self.n_samples)]
                sorted_distances = sorted(distances, key=lambda x:x[0])
                sorted_distances = [x[1] for x in sorted_distances[:k]]
                # print(sorted_distances, i, "\n")
                for j in sorted_distances:
                    self.graph[i, j] = -1
                    self.graph[j, i] = -1

        return self.graph

    def create_cluster(self, k):
        self.k = k
        degree = [sum(x) for x in self.graph] # the degree of each node
        for i in range(self.n_samples): # calculate the graph laplacian
            self.graph[i,i] = -degree[i]

        vals, vecs = np.linalg.eig(self.graph) # calculate eigenvalues and eigenvectors

        # sort these based on the eigenvalues
        vecs = vecs[:, np.argsort(vals)]
        vals = vals[np.argsort(vals)]

        # kmeans on first three vectors with nonzero eigenvalues
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(vecs[:, 1:4])

        self.labels = kmeans.labels_
        return self.labels

    def plot_clusters(self):
        for i, color in zip(range(self.k), 'rgb'):
            same_cluster = (self.labels == i)
            plt.scatter(self.data[same_cluster][:, 0], self.data[same_cluster][:, 1], c=color)
        plt.show()

if __name__ == "__main__":
    n_samples = 1500
    k=2
    # data, label = datasets.make_moons(n_samples=n_samples, noise=0.08)

    n_samples = 1500
    random_state = 170
    data, label = datasets.make_blobs(n_samples=n_samples, cluster_std=[2.5, 1.5, 0.5], random_state=random_state)
    k = 3

    cluster = spectral_clustering(data)
    # cluster.build_graph(neighborhood="in_radius", radius=0.5)
    cluster.build_graph(neighborhood="k_nearest", k=10)
    cluster.create_cluster(k)
    cluster.plot_clusters()
    # cluster.build_graph(neighborhood="k_nearest", k=10)


