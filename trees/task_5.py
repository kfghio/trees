import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('processed_manga_data.csv', header=0)
limited_data = data.head(5713)

target_column = 'status'
X = limited_data.drop(columns=[target_column])
y = limited_data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

boost_train_accuracy = []
boost_test_accuracy = []
n_estimators_values = [1, 2, 5, 10, 15, 20, 30, 40, 50, 100]

for n_estimators in n_estimators_values:
    print(n_estimators)
    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    boost_train_accuracy.append(model.score(X_train, y_train))
    boost_test_accuracy.append(model.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_values, boost_train_accuracy, label='Boosting Train Accuracy', linestyle='--')
plt.plot(n_estimators_values, boost_test_accuracy, label='Boosting Test Accuracy', linestyle='--')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Trees (Boosting)')
plt.legend()
plt.grid()
plt.show()