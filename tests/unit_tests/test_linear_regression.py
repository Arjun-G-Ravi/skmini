import numpy as np
from skmini.regression import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pytest


def test_train_model_make_regression():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LinearRegression(lr=1, num_epochs=100, verbose=True)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    assert type(train_score) == np.float64
    assert type(test_score) == np.float64


def test_train_custom_data():
    X = np.array([[3], [5], [7], [9], [12]])
    y = np.array([1, 2, 3, 4, 5.6])
    model = LinearRegression(num_epochs=100, verbose=True)
    model.fit(X, y)
    prediction = model.predict(X)
    assert type(prediction) == np.ndarray


if __name__ == "__main__":
    test_train_model_make_regression()
    test_train_custom_data()
