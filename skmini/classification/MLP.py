import numpy as np

class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0

    def __mul__(self, other):
        return Value(self.data * other.data)

    def __repr__(self):
        return f'Value({self.data})'

class Operator:
    def __init__(self, data):
        self.data = data
        self.children = []
    
    def __call__(self, x):
        return x

class Neuron:
    def __init__(self):
        self.w = np.random(1,2)
        self.b = np.random(1)







# test -> will change to tests in the future
if __name__ == "__main__":
    # from sklearn.model_selection import train_test_split
    # from sklearn import datasets

    # X, y = datasets.make_classification(
    #     n_samples=100, n_features=2, n_classes=2, n_redundant=0, random_state=42
    # )
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=123
    # )
    # print("X.shape:", X.shape)
    # obj = MLPClassifier()
    # obj.fit(X_train, y_train)

    a = Value(2)
    b = Value(3)
    c = a * b
    print(c)
    op = Operator('*')
    print(op(3))
