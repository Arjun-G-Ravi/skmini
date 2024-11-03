from skmini.classification import LogisticRegression
model = LogisticRegression(verbose=True)
X = [1,2,3,4]
y = [1,1,2,2]

model.fit(X, y)

