# K-Means Clustering From Scratch

An educational implementation of K-Means clustering using NumPy. The project explains the algorithm step by step and compares the custom implementation with scikit-learn on a small, reproducible two-dimensional dataset.

## What this project demonstrates

- Random centroid initialization with a reproducible random seed
- Euclidean distance calculation
- Cluster assignment and centroid updates
- Convergence based on centroid movement
- Empty-cluster handling
- Inertia calculation and training history
- Centroid movement visualization
- The Elbow Method for comparing values of `K`
- Silhouette Score evaluation
- Why feature scaling matters for distance-based algorithms
- Predicting clusters for new observations
- Comparison with `sklearn.cluster.KMeans`
- The effect of random initialization on the final solution

## Project structure

```text
K means Clustering From Scratch/
├── README.md
├── requirements.txt
├── data/
│   └── notebooks/
│       └── K_Means.ipynb
├── src/
│   └── kmeans.py
└── test/
	└── test_kmeans.py
```

## Setup

From this project directory, create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The requirements include NumPy, Matplotlib, and the Jupyter kernel package used by the notebook. The notebook also uses scikit-learn for evaluation and comparison, so install it if it is not already available:

```bash
python -m pip install scikit-learn
```

## Run the notebook

Open `data/notebooks/K_Means.ipynb` in VS Code or Jupyter and run the cells from top to bottom.

The notebook creates a synthetic dataset with three naturally separated groups, trains the custom model, visualizes the results, evaluates several values of `K`, and compares the result with scikit-learn.

## Run the tests

From the project directory:

```bash
python -m pytest test
```

The tests cover fitting, `fit_predict`, prediction after training, invalid input, invalid `K`, and attempting to predict before fitting.

## Using the implementation

```python
import numpy as np

from src.kmeans import KMeans

X = np.array([
	[1.0, 2.0],
	[1.5, 1.8],
	[7.0, 8.0],
	[8.0, 8.5],
])

model = KMeans(
	k=2,
	max_iters=100,
	tolerance=1e-4,
	random_state=42,
)

labels = model.fit_predict(X)

print(labels)
print(model.centroids)
print(model.inertia_)
print(model.n_iter_)

new_points = np.array([
	[1.4, 1.9],
	[7.5, 8.1],
])

print(model.predict(new_points))
```

## `KMeans` parameters

| Parameter | Default | Description |
|---|---:|---|
| `k` | `3` | Number of clusters. Must be positive and no greater than the number of samples. |
| `max_iters` | `100` | Maximum number of optimization iterations. |
| `tolerance` | `1e-4` | Training stops when the total centroid movement is below this value. |
| `random_state` | `None` | Seed for reproducible centroid initialization. |

After fitting, the model exposes:

- `centroids`: learned cluster centers
- `labels`: labels assigned to the training samples
- `inertia_`: final sum of squared distances to the assigned centroids
- `inertia_history`: inertia recorded at each iteration
- `centroid_history`: centroid positions recorded during training
- `n_iter_`: number of iterations completed

## How K-Means works

K-Means alternates between two steps:

1. **Assignment:** assign each sample to its nearest centroid.
2. **Update:** replace each centroid with the mean of its assigned samples.

The process repeats until the centroids move less than the configured tolerance or the maximum number of iterations is reached.

The objective minimized by K-Means is the inertia:

```text
inertia = sum(||x_i - c_label_i||^2)
```

Lower inertia means that samples are closer to their assigned centroids, but inertia generally decreases as `K` increases. It should therefore be considered together with metrics such as the Silhouette Score and with domain knowledge.

## Choosing the number of clusters

The notebook compares values of `K` using:

- **Elbow Method:** look for a point where additional clusters provide diminishing reductions in inertia.
- **Silhouette Score:** prefer values where samples are close to their own cluster and well separated from neighboring clusters. Scores are approximately between `-1` and `1`; higher is generally better.

## Feature scaling

K-Means is distance-based. A feature with a large numeric range can dominate the Euclidean distance, even when it is not more important. The notebook demonstrates standardization with `sklearn.preprocessing.StandardScaler` before clustering.

For real datasets, fit the scaler on the training data and apply the same transformation to future observations before calling `predict`.

## Comparison with scikit-learn

Cluster IDs are arbitrary: label `0` in one implementation may correspond to label `2` in another implementation. The notebook therefore compares inertia, iterations, centroids, and visual cluster structure rather than requiring raw labels to match.

Scikit-learn may produce a better solution because it supports multiple initializations through `n_init`. This implementation intentionally uses one initialization so that the core algorithm remains easy to inspect.

## Limitations and possible improvements

This is a learning implementation, not a production clustering library. It does not currently include:

- K-Means++ initialization
- Multiple random restarts
- Mini-batch training
- Automatic selection of `K`
- Advanced numerical optimizations
- Support for missing values or sparse matrices

Possible next steps include adding K-Means++, comparing K-Means with K-Medoids, and applying the workflow to a real customer-segmentation dataset.

## Learning outcome

The central idea is simple:

```text
Assign -> Update -> Repeat
```

The notebook also shows why practical clustering requires decisions about initialization, feature scaling, the number of clusters, evaluation, and interpretation.
