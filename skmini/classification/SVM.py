import numpy as np
import matplotlib.pyplot as plt


class SVC:
    def __init__(self, kernel='linear'):
        self.kernel = kernel
        assert self.kernel == 'linear', 'Only linear SVM implemented'

    def fit(self, X_train, y_train, n_epochs=10, lr=0.01, lambda_param = 0.01):
        self.n, self.m = X_train.shape # n is num_samples, m is num_features
        self.w = np.array([np.random.randn(1) for i in range(self.m)]).reshape(-1) * 0.01
        self.b = np.random.randn(1) * 0.01
        self.lr = lr
        self.lambda_param = lambda_param
        total_loss = 0
        y_ = np.where(y_train <= 0, -1, 1)

        for i in range(n_epochs):
            for id in range(self.n):
                score = self.w@X_train[id] + self.b
                if (y_[id]* score) >=1:
                    # Correct classification with margin
                    self.w -= self.lr * (2 * self.lambda_param * self.w )
                else:
                    # Incorrect classification or within margin
                    # print((2 * self.lambda_param * self.w).shape, X_train[id].shape, y_[id].shape, )
                    self.w -= self.lr * (2 * self.lambda_param * self.w  - X_train[id]*y_[id])
                    self.b -= self.lr * y_[id]
                # Calculate hinge loss for monitoring
                # print(y_[id], score)
                loss = max(0, 1 - y_[id] * score[0]) 
                print(loss)


        

# if __name__ == '__main__':
#     from skmini.datasets import load_iris
#     # X = load_iris()['data']
#     # y = load_iris()['target']
#     # print(X.shape, y.shape)

#     X = [3,4,5]
#     y = [3,4,5]
#     plt.plot(X, y)
#     plt.show()
#     svm = SVC()
#     svm.fit(X, y)
