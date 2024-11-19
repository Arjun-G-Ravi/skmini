import numpy as np


class SVC:
    def __init__(self, kernel='linear'):
        self.kernel = kernel
        assert self.kernel == 'linear', 'Only linear SVM implemented'

    def fit(self, X_train, y_train, epochs=10, lr=0.01, lambda_param = 0.01):
        self.n, self.m = X_train.shape # n is num_samples, m is num_features
        self.w = np.array([np.random.randn(1) for i in range(self.m)]).reshape(-1) * 0.01
        self.b = np.random.randn(1) * 0.01
        self.lr = lr
        self.lambda_param = lambda_param
        y_ = np.where(y_train <= 0, -1, 1)

        for i in range(epochs):
            epoch_loss = 0
            for id in range(self.n):
                score = self.w@X_train[id] + self.b
                # Calculate hinge loss for monitoring
                loss = max(0, 1 - y_[id] * score[0]) 
                epoch_loss += loss
                if (y_[id]* score) >=1:
                    # Correct classification with margin
                    self.w -= self.lr * (2 * self.lambda_param * self.w )
                else:
                    # Incorrect classification or within margin
                    self.w -= self.lr * (2 * self.lambda_param * self.w  - X_train[id]*y_[id])
                    self.b -= self.lr * y_[id]
            print(epoch_loss)
