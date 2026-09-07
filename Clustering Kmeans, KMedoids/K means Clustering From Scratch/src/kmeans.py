import numpy as np


class KMeans:
    """
    K-Means clustering implementation from scratch using NumPy.

    Parameters
    ----------
    k : int
        Number of clusters.
    max_iters : int
        Maximum number of training iterations.
    tolerance : float
        Training stops when centroid movement is below this value.
    random_state : int or None
        Random seed used for centroid initialization.
    """

    def __init__(
        self,
        k=3,
        max_iters=100,
        tolerance=1e-4,
        random_state=None
    ):
        if not isinstance(k, int):
            raise TypeError("k must be an integer.")

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        if max_iters <= 0:
            raise ValueError("max_iters must be greater than 0.")

        if tolerance < 0:
            raise ValueError("tolerance cannot be negative.")

        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.random_state = random_state

        # Learned parameters
        self.centroids = None
        self.labels = None

        # Training information
        self.inertia_ = None
        self.inertia_history = []
        self.centroid_history = []
        self.n_iter_ = 0

    def _validate_data(self, X):
        """Validate and convert input data."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if len(X) == 0:
            raise ValueError("X cannot be empty.")

        if self.k > len(X):
            raise ValueError(
                "k cannot be greater than the number of samples."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contains NaN or infinite values."
            )

        return X

    def _calculate_distances(self, X):
        """
        Calculate Euclidean distance from every point
        to every centroid.

        Returns
        -------
        distances : np.ndarray
            Shape: (n_samples, k)
        """
        return np.sqrt(
            np.sum(
                (X[:, np.newaxis] - self.centroids) ** 2,
                axis=2
            )
        )

    def _assign_clusters(self, X):
        """Assign every point to its nearest centroid."""
        distances = self._calculate_distances(X)

        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """Calculate new centroids from cluster assignments."""
        new_centroids = np.zeros_like(self.centroids)

        for cluster_id in range(self.k):
            cluster_points = X[labels == cluster_id]

            if len(cluster_points) > 0:
                new_centroids[cluster_id] = cluster_points.mean(
                    axis=0
                )
            else:
                # Keep the old centroid if the cluster is empty.
                new_centroids[cluster_id] = (
                    self.centroids[cluster_id]
                )

        return new_centroids

    def fit(self, X):
        """
        Train the K-Means model.

        Parameters
        ----------
        X : np.ndarray
            Training data with shape (n_samples, n_features).

        Returns
        -------
        self
        """
        X = self._validate_data(X)

        rng = np.random.default_rng(self.random_state)

        
        # 1. Initialize centroids
        
        indices = rng.choice(
            len(X),
            size=self.k,
            replace=False
        )

        self.centroids = X[indices].copy()

        # Reset training history
        self.centroid_history = [
            self.centroids.copy()
        ]

        self.inertia_history = []
        self.n_iter_ = 0

        
        # 2. K-Means optimization loop
        
        for iteration in range(self.max_iters):

            # Assignment step
            labels = self._assign_clusters(X)

            # Calculate inertia using current centroids
            inertia = np.sum(
                (X - self.centroids[labels]) ** 2
            )

            self.inertia_history.append(inertia)

            # Update step
            new_centroids = self._update_centroids(
                X,
                labels
            )

            # Measure centroid movement
            centroid_shift = np.linalg.norm(
                new_centroids - self.centroids
            )

            # Store new centroid positions
            self.centroid_history.append(
                new_centroids.copy()
            )

            # Update centroids
            self.centroids = new_centroids

            self.n_iter_ = iteration + 1

            # Check convergence
            if centroid_shift < self.tolerance:
                break

        
        # 3. Calculate final labels and inertia
        
        self.labels = self._assign_clusters(X)

        self.inertia_ = np.sum(
            (X - self.centroids[self.labels]) ** 2
        )

        return self

    def predict(self, X):
        """
        Assign new samples to the nearest cluster.

        Parameters
        ----------
        X : np.ndarray
            Data to cluster.

        Returns
        -------
        np.ndarray
            Cluster labels.
        """
        if self.centroids is None:
            raise ValueError(
                "Model must be fitted before prediction."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.centroids.shape[1]:
            raise ValueError(
                "X has a different number of features "
                "than the training data."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contains NaN or infinite values."
            )

        return self._assign_clusters(X)

    def fit_predict(self, X):
        """
        Train the model and return cluster labels.
        """
        self.fit(X)

        return self.labels

    def inertia(self, X):
        """
        Calculate inertia for a dataset using the
        currently fitted centroids.
        """
        if self.centroids is None:
            raise ValueError(
                "Model must be fitted before calculating inertia."
            )

        X = np.asarray(X, dtype=float)

        labels = self.predict(X)

        return np.sum(
            (X - self.centroids[labels]) ** 2
        )
