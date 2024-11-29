import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from RF import RandomForestClassifier as RandomForestClassifier2
from sklearn.ensemble import RandomForestClassifier

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

n_estimators_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
library_train_accuracy_rf = []
library_test_accuracy_rf = []
custom_train_accuracy_rf = []
custom_test_accuracy_rf = []

for n_estimators in n_estimators_values:
    print(n_estimators)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    library_train_accuracy_rf.append(model.score(X_train, y_train))
    library_test_accuracy_rf.append(model.score(X_test, y_test))

    custom_forest = RandomForestClassifier2(n_estimators=n_estimators)
    custom_forest.fit(X_train_np, y_train_np)
    custom_train_preds = custom_forest.predict(X_train_np)
    custom_test_preds = custom_forest.predict(X_test_np)
    custom_train_accuracy_rf.append(accuracy_score(y_train_np, custom_train_preds))
    custom_test_accuracy_rf.append(accuracy_score(y_test_np, custom_test_preds))

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_values, library_train_accuracy_rf, label='Library Train Accuracy', linestyle='--')
plt.plot(n_estimators_values, library_test_accuracy_rf, label='Library Test Accuracy', linestyle='--')
plt.plot(n_estimators_values, custom_train_accuracy_rf, label='Custom Train Accuracy', linestyle='-')
plt.plot(n_estimators_values, custom_test_accuracy_rf, label='Custom Test Accuracy', linestyle='-')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Trees (Library and Custom)')
plt.legend()
plt.grid()
plt.show()