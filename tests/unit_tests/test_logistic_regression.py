# set the correct path
import set_path

from skmini.classification import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
ds = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(ds['data'], ds['target'])

def test_log_regression():
    model = LogisticRegression()
    model.fit(X_train, y_train)

    assert type(model.score(X_train, y_train)) == float, 'The accuracy should be a float'
    print(model.score(X_test, y_test))

if __name__ == '__main__':
    test_log_regression()