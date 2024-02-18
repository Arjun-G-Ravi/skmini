import numpy as np
from copy import deepcopy

class MLPClassifier:
    def __init__(self,lr=0.01, num_epochs=100, layers=[3,2,1]):
        self.lr = lr
        self.num_epochs = num_epochs
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = [] # Can't initialize now, as X.shape is not available yet
        

    def fit(self, X, y):
        m, n = X.shape # num_data X num_features
        y = np.array(y, dtype=np.float32).reshape(-1)
        next_weight = n
        for i in self.layers:
            w = np.random.randn(next_weight, i) 
            self.weights.append(deepcopy(w))
            next_weight = i    
            
        # forward pass
        out = X
        for w in self.weights[:-1]:
            out = self.relu(out@w)
        out = self.sigmoid(out@self.weights[-1])
        y_pred = np.array(out > 0.5, dtype=np.float32).reshape(-1)

        # define binary cross entropy loss
        loss = -np.mean(y*np.log(y_pred+1e-8) + (1-y)*np.log(1-y_pred+1e-8))
        print(loss)
        
        # backpropogation
        
        

        return 

    def predict(self, X):
        out = X
        for w in self.weights[:-1]:
            out = self.relu(out@w)
        out = self.sigmoid(out@self.weights[-1])
        y_pred = np.array(out > 0.5, dtype=np.float32).reshape(-1)
        return y_pred
    
    def relu(self,X):
        return np.maximum(0, X)

    def sigmoid(self, x):
        out = 1/(1+np.exp(-x))
    
        # print(out.shape)
        return out

# test
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    X, y = datasets.make_classification(n_samples=100, n_features=2, n_classes=2, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    print('X.shape:',X.shape)
    obj = MLPClassifier()
    obj.fit(X_train, y_train)
    