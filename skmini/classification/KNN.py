import numpy as np


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def _predict_one(self, x):
        neighbours = []
        assert x.shape == self.X[0].shape
        for x_, y_ in zip(self.X, self.y):
            neighbours.append((self._calc_distance(x, x_), y_))
        neighbours.sort()
        top_k_nearest_neighbours = neighbours[: self.k] # top-k neighbours
        score = 0
        y_classes = set(self.y)
        data = {k:0 for k in y_classes}
        for i in top_k_nearest_neighbours:
            data[i[-1]] += 1
        return max(data, key=data.get)

    def _calc_distance(self, x1, x2):
        return (np.sum((x1 - x2)**2))**0.5


if __name__ == "__main__":

    X = np.array(
        [[1, 2], [2, 3], [3, 3], [6, 6], [7, 8], [8, 8], [1, 0], [0, 1], [7, 7], [2, 2]]
    )
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0])
    knn = KNN(5)
    knn.fit(X, y)
    print(knn._predict_one(np.array([6, 7])))
