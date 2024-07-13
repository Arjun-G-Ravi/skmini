import numpy as np

class Value:
    def __init__(self, data, _op = '', _children = []):
        self.data = data
        self.grad = None
        self.op = _op
        self.children = _children
        self.backward = None

    def __mul__(self, other):
        if not isinstance(other, Value): other = Value(other)
        out = Value(self.data * other.data, _op = '*', _children = [self, other])

        def _backward():
            self.grad = other.data
            other.grad = self.data
        out.backward = _backward
        return out

    def __add__(self, other):
        if not isinstance(other, Value): other = Value(other)
        out = Value(self.data+other.data, _op = '+', _children = [self, other])

        def _backward():
            self.grad = out.grad
            other.grad = out.grad
            
        out.backward = _backward
        return out

    def relu(self):
        out = Value(max(self.data, 0), _op = 'relu', _children = [self])

        def _backward():
            if out == 0: self.grad = 0
            else: self.grad = 1
        out.backward = _backward
        return out


    def __repr__(self):
        return f'Value(data:{self.data} grad:{self.grad})\n'
    
    def backprop(self):
        # traverses through the tree, till it reaches the leaf nodes AND applies gradient to each of the nodes.
        self.backward()

        for child in self.children:
            if child.op:
                child.backprop()

class Neuron:
    def __init__(self):
        self.w = None
        self.b = None
        self.parameters = [self.w, self.b]
    
    def __repr__(self):
        return f'Neuron {self.w, self.b}'

class Layer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.data = [Neuron() for i in range(self.num_neurons)]
        self.parameters = [n.parameters for n in self.data]


    def __repr__(self):
        return f'Layer {self.data}'

class MLP:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.data = []
        for layer in (layers):
            self.data.append(Layer(layer))

        self.parameters = [l.parameters for l in self.data]

        for p in self.parameters:
            print(p)
        # print(self.data)
    
    def __repr__(self):
        return f'MLP {self.data}'

    def fit(self, X, y):
        m, n = X.shape
        # if not earlier defined, randomly initialize the weights

        # forward pass
        # loss calc
        # backward pass

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
    b = Value(4)
    c = a + b
    d = Value(5)
    e = c * d
    f = e.relu()
    f.grad = 1
    f.backprop()
    print(a, b, c, d, e, f)

    MLP([5,3,1])

