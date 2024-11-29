import numpy as np


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _split(self, X_column, threshold):
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        return left_mask, right_mask

    def _best_split(self, X, y):
        best_gini = float("inf")
        best_split = None

        for column_index in range(X.shape[1]):
            thresholds = np.unique(X[:, column_index])
            for threshold in thresholds:
                left_mask, right_mask = self._split(X[:, column_index], threshold)

                if sum(left_mask) < self.min_samples_leaf or sum(right_mask) < self.min_samples_leaf:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                gini_split = (sum(left_mask) * gini_left + sum(right_mask) * gini_right) / len(y)

                if gini_split < best_gini:
                    best_gini = gini_split
                    best_split = {
                        "feature": column_index,
                        "threshold": threshold,
                        "gini": gini_split,
                    }
        return best_split

    def _majority_class(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def _build_tree(self, X, y, depth=0):

        if self.max_depth is not None and depth >= self.max_depth:
            return {"value": self._majority_class(y)}

        if len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return {"value": self._majority_class(y)}

        best_split = self._best_split(X, y)
        if best_split is None:
            return {"value": self._majority_class(y)}

        feature = best_split["feature"]
        threshold = best_split["threshold"]

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        if sum(left_indices) < self.min_samples_leaf or sum(right_indices) < self.min_samples_leaf:
            return {"value": self._majority_class(y)}

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left_indices], y[left_indices], depth + 1),
            "right": self._build_tree(X[right_indices], y[right_indices], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_one(self, x, tree):
        if "value" in tree:
            return tree["value"]

        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_one(x, tree["left"])
        else:
            return self._predict_one(x, tree["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])
