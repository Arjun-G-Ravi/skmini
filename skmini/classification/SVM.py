import numpy as np
import matplotlib.pyplot as plt


class SVC:
    def __init__(self, kernel='linear'):
        self.kernel = kernel

    def fit(self, X_train, y_train):
        self.dim = X_train.shape[-1] - 1
        self.w = [np.random.randn(1) for i in range(self.dim-1)]
        self.b = np.random.randn(1)
        

        print(self.w)
        print(X_train)
        y = np.arange(-10, 11, 10)
        print(y)

        plt.plot((X_train*self.w).reshape(3,), y)
        plt.scatter(X_train,y_train)
    
    def _distance(self, x):
        return 1

        

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
