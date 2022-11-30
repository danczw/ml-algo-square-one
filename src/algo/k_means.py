import numpy as np
import matplotlib.pyplot as plt


class KMeans():
    def __init__(self, K:int=3, max_iters:int=100, plot_steps:bool=False):
        self.name = "K-Means"
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # the centers (mean vector) for each cluster
        self.centroids = []
    
    # calculate euclidean distance between two vectors
    def _euclidean_distance(self, x1:np.ndarray, x2:np.ndarray) -> float:
        return np.sqrt(np.sum((x1-x2)**2))

    # predict cluster for each sample
    def predict(self, X:np.ndarray) -> np.ndarray:
        self.X = X
        self.n_samples, self.n_features = X.shape

        # randomly initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimize clusters
        for _ in range(self.max_iters):
            # assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    # get cluster labels for each sample
    def _get_cluster_labels(self, clusters:np.ndarray) -> np.ndarray:
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    # assign samples to closest centroids
    def _create_clusters(self, centroids:np.ndarray) -> np.ndarray:
        # assign the samples to the closest centroids
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    # calculate distance to all centroids
    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [self._euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    # calculate new centroids as the means of the samples in each cluster
    def _get_centroids(self, clusters:np.ndarray) -> np.ndarray:
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    # check if centroids have changed, if not, the algorithm has converged
    def _is_converged(self, centroids_old, centroids):
        # distances between old and new centroids, for all centroids
        distances = [self._euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
    
    # plot clusters and centroids
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.savefig('k_means_cluster.png')