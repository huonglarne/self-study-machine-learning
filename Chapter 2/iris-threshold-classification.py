import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']

# Plot distribution by pair of features
for t, marker,c in zip(range(3), '>ox', 'rbg'):
    plt.scatter(features[target == t, 0], features[target == t, 2], marker=marker, c=c)
plt.show()

# Identify setosa (target 0) separated by petal length (feature 2)
plength = features[:, 2]
is_setosa = (target == 0)
max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()
print(max_setosa, min_non_setosa)

# Identify virginica (target 1) by selecting the features that has the highest accuracy to separate
non_setosa_features = features[~is_setosa]
non_setosa_target = target[~is_setosa]
is_virginica = non_setosa_target[non_setosa_target == 1]

best_acc = -1.0
for fi in range(non_setosa_features.shape[1]):
    thresh = non_setosa_features[:, fi].copy()
    thresh.sort()

    for t in thresh:
        pred = (non_setosa_features[:, fi] > t)
        acc = (pred == 1).mean()
        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t


