import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def _predict_one(self, x):
        neighbours = []
        for x_, y_ in zip(self.X, self.y):
            # print(i, x)
            neighbours.append((self._calc_distance(x, x_), y_))
        print(neighbours)
        neighbours.sort()
        print(neighbours) 
        k_nearest_neighbours = neighbours[:self.k]
        score = 0
        for i in k_nearest_neighbours:
            if i[-1] == 1: score += 1
            else: score -= 1
        if score >=0: return 1
        else: return 0



    def _calc_distance(self, x1, x2):
        return np.sum(np.abs(x1-x2))



if __name__ == "__main__":

    X = np.array([[1, 2], [2, 3], [3, 3], [6, 6], [7, 8], [8, 8], [1, 0], [0, 1], [7, 7], [2, 2]])
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0])
    knn = KNN(5)
    knn.fit(X, y)
    print(knn._predict_one(np.array([6,7])))