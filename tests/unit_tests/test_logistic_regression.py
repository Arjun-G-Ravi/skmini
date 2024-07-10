import numpy as np
from skmini.classification import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class TestLogisticRegression:
    def test_train_model_breastcancer(self):
        ds1 = load_breast_cancer()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            ds1["data"], ds1["target"]
        )
        model = LogisticRegression(verbose=True, num_epochs=1000)

        model.fit(self.X_train, self.y_train)
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)
        assert type(train_score) == np.float64
        assert type(test_score) == np.float64
        print(train_score, test_score)

    def test_train_custom_data(self):
        X = np.array([[3], [5], [7], [9], [12]])
        y = np.array([0, 0, 1, 1, 1])
        model = LogisticRegression(num_epochs=10000,verbose=True )
        model.fit(X, y)
        train_score = model.score(X,y)
        prediction = model.predict(X)
        assert type(prediction) == np.ndarray
        print(train_score)


if __name__ == "__main__":
    f = TestLogisticRegression()
    # f.test_train_model_breastcancer()
    # print(' - '*40)
    f.test_train_custom_data()
