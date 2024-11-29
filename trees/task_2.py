import pandas as pd
from sklearn.model_selection import train_test_split
from DT import DecisionTreeClassifier
import matplotlib.pyplot as plt



data = pd.read_csv('processed_manga_data.csv', header=0)
limited_data = data.head(500)

target_column = 'status'
X = limited_data.drop(columns=[target_column])
y = limited_data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values
y_test_np = y_test.values

tree_depths_custom = []
min_samples_leaf_values = [1, 5, 10, 20, 50]

for min_samples_leaf in min_samples_leaf_values:
    print(min_samples_leaf)
    custom_tree = DecisionTreeClassifier(max_depth=None, min_samples_leaf=min_samples_leaf)
    print(1)
    custom_tree.fit(X_train_np, y_train_np)
    print(2)

    def calculate_depth(tree, depth=0):
        if "value" in tree:
            return depth
        return max(calculate_depth(tree["left"], depth + 1), calculate_depth(tree["right"], depth + 1))

    tree_depths_custom.append(calculate_depth(custom_tree.tree))

# Построение графика
plt.figure(figsize=(8, 5))
plt.plot(min_samples_leaf_values, tree_depths_custom, marker='o', label='Custom Implementation', color='orange')
plt.xlabel('Min Samples Leaf')
plt.ylabel('Tree Depth')
plt.title('Dependency of Tree Depth on Hyperparameters (Custom)')
plt.legend()
plt.grid()
plt.show()