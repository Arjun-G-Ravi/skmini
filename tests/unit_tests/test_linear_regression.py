import unittest
import numpy as np
from skmini.regression import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Code to test linear regression is not done. This is just the one for logistic.


class TestLinearRegression(unittest.TestCase):
    def test_train_model_breastcancer(self):
        ds1 = make_regression(10)
        print(ds1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            ds1["data"], ds1["target"]
        )

        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)
        self.assertIsInstance(
            train_score, np.float64
        ), "The accuracy should be a NumPy array"
        self.assertIsInstance(
            test_score, np.float64
        ), "The accuracy should be a NumPy array"

    # def test_train_custom_data(self):
    #     X = np.array([[3], [5], [7], [9], [12]])
    #     y = np.array([0, 0, 1, 1, 1])
    #     model = LogisticRegression(num_epochs=200)
    #     model.fit(X, y)
    #     prediction = model.predict(X)
    #     self.assertIsInstance(
    #         prediction, np.ndarray
    #     ), "The accuracy should be a NumPy array"


if __name__ == "__main__":
    unittest.main()
