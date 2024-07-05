import unittest
import numpy as np
from skmini.regression import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


class TestLinearRegression(unittest.TestCase):
    def test_train_model_make_regression(self):
        X, y = make_regression(n_samples=100, n_features=10, noise=0.1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
        model = LinearRegression(lr=0.1, num_epochs=500)
        model.fit(self.X_train, self.y_train)
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)
        self.assertIsInstance(
            train_score, np.float64
        ), "The accuracy should be a NumPy array"
        self.assertIsInstance(
            test_score, np.float64
        ), "The accuracy should be a NumPy array"

    def test_train_custom_data(self):
        X = np.array([[3], [5], [7], [9], [12]])
        y = np.array([1, 2, 3, 4, 5.6])
        model = LinearRegression(num_epochs=200)
        model.fit(X, y)
        prediction = model.predict(X)
        self.assertIsInstance(
            prediction, np.ndarray
        ), "The accuracy should be a NumPy array"


if __name__ == "__main__":
    unittest.main()
