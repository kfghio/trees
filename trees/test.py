import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from DT import DecisionTreeClassifier
from RF import RandomForestClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv('processed_manga_data.csv', header=0)
limited_data = data.head(5713)

target_column = 'status'
X = limited_data.drop(columns=[target_column])
y = limited_data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values
y_test_np = y_test.values

tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1)
tree.fit(X_train_np, y_train_np)
tree_predictions = tree.predict(X_test_np)

#custom_tree1 = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1)
#custom_tree1.fit(X_train_np, y_train_np)

#custom_tree2 = DecisionTreeClassifier(max_depth=50, min_samples_split=2, min_samples_leaf=1)
#custom_tree2.fit(X_train_np, y_train_np)

#print("Accuracy Tree 1:", np.mean(custom_tree1.predict(X_test_np) == y_test_np))
#print("Accuracy Tree 2:", np.mean(custom_tree2.predict(X_test_np) == y_test_np))

forest = RandomForestClassifier(n_estimators=10, max_depth=5, min_samples_split=10, min_samples_leaf=5)
forest.fit(X_train_np, y_train_np)
forest_predictions = forest.predict(X_test_np)

tree_accuracy = accuracy_score(y_test_np, tree_predictions)
forest_accuracy = accuracy_score(y_test_np, forest_predictions)

print(f"Accuracy дерева решений: {tree_accuracy:.4f}")
print(f"Accuracy случайного леса: {forest_accuracy:.4f}")