from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 0, 1, 1]


rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)
clf = SGDClassifier()
clf.fit(X_features, y)





print(clf.predict(X_features))