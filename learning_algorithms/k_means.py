import numpy as np

from interfaces.base_model import BaseModel

def k_means_train(X, k, iteration_count=1000):
    centroids = X[np.random.choice(X.shape[0], k)] # replacement # k*n

    for i in range(iteration_count):
        # Partition: assign points to the closet centroids
        distance = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2) # m*1*n k*n => m*k
        closest_centroids_ids = np.atleast_2d(np.argmin(distance, axis=1)).T # m*1

        # Move centroids: assign centroids to the cluster center
        centroids_ids, counts = np.unique(closest_centroids_ids, return_counts=True)
        centroids_ids, counts = np.atleast_2d(centroids_ids), np.atleast_2d(counts) # 1*c, 1*c
        centroids_indicator = closest_centroids_ids == centroids_ids # m*c
        centroids = (centroids_indicator.T @ X) / counts.T

        # re-pick centroids with no point
        single_centroids_ids = counts == 0
        single_centroids_count = np.sum(single_centroids_ids)
        if single_centroids_count > 0:
            centroids[single_centroids_ids] = X[np.random.choice(X.shape[0], single_centroids_count)]

    return centroids


def k_means_predict(X, centroids):
    distance = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    centroids_ids = np.atleast_2d(np.argmin(distance, axis=1)).T
    return centroids[centroids_ids]


def k_means_loss(X, Y):
    return np.mean(np.linalg.norm(X - Y, axis=1, keepdims=True))


class KMeans (BaseModel):

    def learn(self, X, k, iteration_count=1000):
        self.centroids = k_means_train(X, k, iteration_count)

    def infer(self, X):
        return k_means_predict(X, self.centroids)
