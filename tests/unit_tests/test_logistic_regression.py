import set_path  # sets the correct path

import numpy as np
from skmini.classification import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

ds1 = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(ds1["data"], ds1["target"])


def test_log_regression():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    assert isinstance(train_score, np.float64), "The accuracy should be a NumPy array"
    assert isinstance(test_score, np.float64), "The accuracy should be a NumPy array"


if __name__ == "__main__":
    test_log_regression()
