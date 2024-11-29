import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

from sklearn.tree import DecisionTreeClassifier

min_samples_leaf_values = [1, 5, 10, 20, 50]
tree_depths_library = []

for min_samples_leaf in min_samples_leaf_values:
    model = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=42)
    model.fit(X_train_np, y_train_np)
    tree_depths_library.append(model.get_depth())

plt.figure(figsize=(8, 5))
plt.plot(min_samples_leaf_values, tree_depths_library, marker='o', label='Library Implementation')
plt.xlabel('Min Samples Leaf')
plt.ylabel('Tree Depth')
plt.title('Dependency of Tree Depth on Hyperparameters (Library)')
plt.legend()
plt.grid()
plt.show()