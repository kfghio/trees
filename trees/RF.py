import numpy as np
from DT import DecisionTreeClassifier
#from sklearn.trees import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", oob_score=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.oob_score = oob_score
        self.trees = []
        self.features = []
        self.oob_score_ = None

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        oob_indices = np.setdiff1d(np.arange(n_samples), indices) if self.oob_score else None
        return X[indices], y[indices], oob_indices

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.trees = []
        self.features = []
        if self.oob_score:
            self.oob_predictions = np.zeros((n_samples, len(self.classes_)))

        for _ in range(self.n_estimators):
            X_sample, y_sample, oob_indices = self._bootstrap_sample(X, y)

            prop = np.random.uniform(0.5, 1.0)
            n_select = max(1, int(prop * n_features))
            selected_features = np.random.choice(n_features, n_select, replace=False)
            self.features.append(selected_features)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_sample[:, selected_features], y_sample)
            self.trees.append(tree)

            if self.oob_score and oob_indices is not None:
                oob_preds = tree.predict_proba(X[oob_indices][:, selected_features])
                for i, idx in enumerate(oob_indices):
                    self.oob_predictions[idx] += oob_preds[i]

        if self.oob_score:
            oob_pred_classes = np.argmax(self.oob_predictions, axis=1)
            self.oob_score_ = np.mean(oob_pred_classes == y)

    def predict(self, X):
        n_classes = len(self.classes_)
        class_probabilities = np.zeros((X.shape[0], n_classes))

        for i, tree in enumerate(self.trees):
            selected_features = self.features[i]
            proba = tree.predict_proba(X[:, selected_features])

            if proba.shape[1] < n_classes:
                proba_padded = np.zeros((proba.shape[0], n_classes))
                proba_padded[:, :proba.shape[1]] = proba
                proba = proba_padded

            class_probabilities += proba

        final_predictions = np.argmax(class_probabilities, axis=1)
        return final_predictions

    def predict_proba(self, X):
        n_classes = len(self.classes_)
        class_probabilities = np.zeros((X.shape[0], n_classes))

        for i, tree in enumerate(self.trees):
            selected_features = self.features[i]
            proba = tree.predict_proba(X[:, selected_features])

            if proba.shape[1] < n_classes:
                proba_padded = np.zeros((proba.shape[0], n_classes))
                proba_padded[:, :proba.shape[1]] = proba
                proba = proba_padded

            class_probabilities += proba

        return class_probabilities / len(self.trees)
