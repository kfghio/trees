import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from DT import DecisionTreeClassifier as DecisionTreeClassifier2

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


library_train_accuracy = []
library_test_accuracy = []
custom_train_accuracy = []
custom_test_accuracy = []

max_depth_values = range(1, 21)

for max_depth in max_depth_values:
    print(max_depth)

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    library_train_accuracy.append(model.score(X_train, y_train))
    library_test_accuracy.append(model.score(X_test, y_test))

    custom_tree = DecisionTreeClassifier2(max_depth=max_depth)
    custom_tree.fit(X_train_np, y_train_np)
    custom_train_preds = custom_tree.predict(X_train_np)
    custom_test_preds = custom_tree.predict(X_test_np)

    custom_train_accuracy.append(accuracy_score(y_train_np, custom_train_preds))
    custom_test_accuracy.append(accuracy_score(y_test_np, custom_test_preds))

print(custom_test_accuracy)

plt.figure(figsize=(10, 6))
plt.plot(max_depth_values, library_train_accuracy, label='Library Train Accuracy', linestyle='--')
plt.plot(max_depth_values, library_test_accuracy, label='Library Test Accuracy', linestyle='--')
plt.plot(max_depth_values, custom_train_accuracy, label='Custom Train Accuracy', linestyle='-')
plt.plot(max_depth_values, custom_test_accuracy, label='Custom Test Accuracy', linestyle='-')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Tree Depth (Library and Custom)')
plt.legend()
plt.grid()
plt.show()