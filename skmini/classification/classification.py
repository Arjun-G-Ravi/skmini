import numpy as np


class LogisticRegression:
    """
    Parameters: lr, num_epochs
    Methods: fit, predict, score

    """

    def __init__(self, lr=0.01, num_epochs=1000):
        self.lr = lr
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = (
            X.shape
        )  # m is the number of training data and n is the number of features
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.num_epochs):
            y_pred = self.predict(X)

            dJ_dw = (1 / m) * np.dot(y_pred - y, X)
            dJ_db = (1 / m) * np.sum(y_pred - y)

            self.weights -= self.lr * dJ_dw
            self.bias -= self.lr * dJ_db

    def predict(self, X):
        out1 = [self._predict_one(x) for x in X]
        out = [1 if i > 0.5 else 0 for i in out1]
        return out

    def score(self, X, y):
        return (1 / len(y)) * np.sum(self.predict(X) == y)

    def _predict_one(self, x):
        return self._sigmoid(
            np.dot(self.weights, x) + self.bias
        )  # f(x) = sigmoid(w*x + b)

    def _sigmoid(self, z):
        # z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-(np.clip(z, -500, 500))))
