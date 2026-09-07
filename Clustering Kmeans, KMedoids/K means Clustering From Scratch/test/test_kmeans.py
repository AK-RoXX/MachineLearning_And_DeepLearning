import numpy as np
import pytest

from src.kmeans import KMeans


def create_dataset():
    return np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [8.0, 8.0],
        [9.0, 8.0],
        [8.0, 9.0],
    ])


def test_kmeans_fit():

    X = create_dataset()

    model = KMeans(
        k=2,
        random_state=42
    )

    model.fit(X)

    assert model.centroids.shape == (2, 2)
    assert model.labels.shape == (6,)
    assert model.inertia_ >= 0
    assert model.n_iter_ > 0


def test_fit_predict():

    X = create_dataset()

    model = KMeans(
        k=2,
        random_state=42
    )

    labels = model.fit_predict(X)

    assert len(labels) == len(X)


def test_predict_after_training():

    X = create_dataset()

    model = KMeans(
        k=2,
        random_state=42
    )

    model.fit(X)

    new_points = np.array([
        [1.5, 1.5],
        [8.5, 8.5]
    ])

    predictions = model.predict(new_points)

    assert predictions.shape == (2,)


def test_predict_before_fit():

    model = KMeans(k=2)

    X = np.array([
        [1, 2],
        [3, 4]
    ])

    with pytest.raises(ValueError):
        model.predict(X)


def test_invalid_k():

    with pytest.raises(ValueError):
        KMeans(k=0)


def test_k_larger_than_samples():

    X = np.array([
        [1, 2],
        [3, 4]
    ])

    model = KMeans(
        k=3,
        random_state=42
    )

    with pytest.raises(ValueError):
        model.fit(X)


def test_invalid_input_dimension():

    model = KMeans(k=2)

    X = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        model.fit(X)
