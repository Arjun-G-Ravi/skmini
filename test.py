from skmini.classification.classification import LogisticRegression

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
ds = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(ds['data'], ds['target'])

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))