import numpy as np
from sklearn import linear_model
X = np.array([[-1,-1], [-2,-1], [1,1], [2,1]])
Y = np.array([-2,-3,2,3])

clf = linear_model.SGDClassifier()
clf.fit(X,Y)

X_score = np.array([[-1, -2], [1,2]])
Y_score = np.array([-2, 2])

print(clf.predict([[-0.8, -1], [1,2]]))
print(clf.score(X_score, Y_score))