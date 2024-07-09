import numpy as np
from skmini.regression import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


class TestLinearRegression:
    def test_train_model_make_regression(self):
        X, y = make_regression(n_samples=100, n_features=10, noise=0.1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
        model = LinearRegression(lr=1, num_epochs=100, verbose=True)
        model.fit(self.X_train, self.y_train)
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)

        assert type(train_score) == np.float64
        assert type(test_score) == np.float64

    def test_train_custom_data(self):
        X = np.array([[3], [5], [7], [9], [12]])
        y = np.array([1, 2, 3, 4, 5.6])
        model = LinearRegression(num_epochs=100, verbose=True)
        model.fit(X, y)
        prediction = model.predict(X)

        assert type(prediction) == np.ndarray


if __name__ == "__main__":
    f = TestLinearRegression()
    # f.test_train_model_make_regression()
    f.test_train_custom_data()
