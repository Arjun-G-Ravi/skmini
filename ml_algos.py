import numpy as np
class ml_algos: # just a sample, mainly for docs
    def __init__(self):
        print("Thank you for using ml_algos")

    def docs(self):
        print( """The current version contains the following implemented algorithms:
            1. Linear Regression # fit, predict, score
            2. Logistic Regression""")
        

class LinearRegression:
    def __init__(self, lr=0.01, num_epochs=100):
        self.lr = lr
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        m, n = X.shape # m is the number of training data and n is the number of features
        self.weights = np.zeros(n)
        self.bias = 0

        for epoch in range(self.num_epochs):
            y_pred = self.predict(X)
            mse_cost = (1/m)*np.sum((y_pred-y)**2) # Cost function J
            print(mse_cost)

            dJ_dw = (1/m)*np.dot(y_pred-y,X)
            dJ_db = (1/m)*np.sum(y_pred-y)

            self.weights -= self.lr * dJ_dw
            self.bias -= self.lr * dJ_db
        
    def predict(self, X):
        return [self.predict_one_(x) for x in X]
        

    def predict_one_(self, x):
        # f(x) = w*x + b
        return np.dot(self.weights,x) + self.bias