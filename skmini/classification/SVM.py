import numpy as np


class SVC:
    def __init__(self, kernel='linear'):
        self.kernel = kernel
        assert self.kernel == 'linear', 'Only linear SVM implemented'
        print('Only binary classification supported in SVM')

    def fit(self, X_train, y_train, epochs=10, lr=0.001, lambda_param = 0.01, bs=64, verbose=True):
        self.bs = bs
        self.n, self.m = X_train.shape # n is num_samples, m is num_features
        self.w = np.array([np.random.randn(1) for i in range(self.m)]).reshape(-1) * 0.1
        self.b = np.random.randn(1) * 0.01
        self.lr = lr
        self.lambda_param = lambda_param
        y_ = np.where(y_train <= 0, -1, 1)

        for _ in range(epochs):
            # ideally, i should not iterate over each of the x value and update w,b with it
            epoch_loss = 0
            for _ in range(self.bs):
                id = np.random.randint(self.n)
                # print(id)
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
            if verbose: print(epoch_loss)
    
    def _predict_one(self, _x, _y):
        score = self.w@_x + self.b
        if score >0: return 1
        else: return 0

    def score(self, X, y):
        correct = 0
        for x_, y_ in zip(X, y):
            y_pred = self._predict_one(x_, y_)
            correct += int(y_ == y_pred)
        return correct/len(y)
