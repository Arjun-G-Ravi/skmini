import numpy as np


class LinearRegression:
    """
    Parameters: lr, num_epochs, show_cost, show_cost_graph
    Methods: fit, predict, score

    """

    def __init__(self, lr=0.01, num_epochs=100, verbose=False):
        self.lr = lr
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = (
            X.shape
        )  # m is the number of training data and n is the number of features
        self.weights = np.zeros(n)
        self.bias = 0
        self.loss_per_epoch = []

        for epoch in range(self.num_epochs):
            y_pred = self.predict(X)
            mse_cost = (1 / m) * np.sum((y_pred - y) ** 2)  # Cost function J
            if self.verbose:
                print(f"Epoch={epoch+1}, MSE={mse_cost}")
            self.loss_per_epoch.append(mse_cost)

            dJ_dw = (1 / m) * np.dot(y_pred - y, X)
            dJ_db = (1 / m) * np.sum(y_pred - y)

            self.weights -= self.lr * dJ_dw
            self.bias -= self.lr * dJ_db

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def score(self, X, y):
        # last mse error
        return self.loss_per_epoch[-1]

    def _predict_one(self, x):
        return np.dot(self.weights, x) + self.bias


if __name__ == "__main__":
    X = 1
    model = LinearRegression()
