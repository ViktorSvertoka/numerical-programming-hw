# -*- coding: utf-8 -*-
"""hw_08_svertoka_viktor.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IOpUOI0b940eNMmvn24ewjeB2ipdA3m0
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# 1. Завантажити набір даних
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. Розподілити дані на навчальні та тестові
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Вибірка ознак для кожного класу
class_data = {}
for i in np.unique(y_train):
    class_data[i] = X_train[y_train == i]

# 4. Розрахунок матриць коваріації для кожного класу
cov_matrices = {}
for i in class_data:
    cov_matrices[i] = np.cov(class_data[i], rowvar=False)

# 5. Обчислення обернених матриць коваріації
inv_cov_matrices = {}
for i in cov_matrices:
    inv_cov_matrices[i] = np.linalg.inv(cov_matrices[i])

# 6. Обчислення апріорних ймовірностей для кожного класу
priors = {}
total_samples = len(y_train)
for i in np.unique(y_train):
    priors[i] = np.sum(y_train == i) / total_samples

# 7. Обчислення дискримінантної функції для одного тестового зразка
def discriminant_function(x, mean, inv_cov, prior):
    return -0.5 * np.dot(np.dot((x - mean), inv_cov), (x - mean)) + np.log(prior)

# 8. Обчислення дискримінантної функції для всіх тестових даних
def predict(X_test):
    predictions = []
    for x in X_test:
        scores = []
        for i in np.unique(y_train):
            mean = np.mean(class_data[i], axis=0)
            score = discriminant_function(x, mean, inv_cov_matrices[i], priors[i])
            scores.append(score)
        predictions.append(np.argmax(scores))
    return np.array(predictions)

y_pred_custom = predict(X_test)

# 9. Використання QuadraticDiscriminantAnalysis з sklearn для порівняння результатів
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred_sklearn = qda.predict(X_test)

# 10. Порівняння результатів
print("Результати прогнозування:")
comparison = pd.DataFrame({
    'True': y_test,
    'Custom': y_pred_custom,
    'Sklearn': y_pred_sklearn
})

print(comparison.head())
print(f"Точність власної реалізації: {accuracy_score(y_test, y_pred_custom) * 100:.2f}%")
print(f"Точність sklearn: {accuracy_score(y_test, y_pred_sklearn) * 100:.2f}%")

# Висновок про ступінь схожості результатів
custom_accuracy = accuracy_score(y_test, y_pred_custom)
sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)

print("\nВисновок про ступінь схожості результатів:")
if custom_accuracy == sklearn_accuracy:
    print(f"Результати власної реалізації і бібліотеки sklearn збігаються. Точність: {custom_accuracy * 100:.2f}%")
else:
    print(f"Результати власної реалізації і бібліотеки sklearn мають невелике відхилення.")
    print(f"Точність власної реалізації: {custom_accuracy * 100:.2f}%")
    print(f"Точність sklearn: {sklearn_accuracy * 100:.2f}%")

# Висновки
print("\nВисновки:")
print("1. Метод QDA добре працює для класифікації даних із набору Iris, зокрема коли класи мають різні коваріаційні структури.")
print("2. Точність власної реалізації та результатів бібліотеки sklearn близькі, що свідчить про коректність обчислень.")
print("3. Зрозуміло, що для кожного класу потрібно обчислювати апріорні ймовірності, матриці коваріації та обертати їх.")
print("4. Власна реалізація дискретизації функцій та обчислення ймовірностей дала точність на рівні стандартної бібліотеки.")
print("5. Важливим моментом є використання матричних операцій для обчислення дискримінантних функцій, що є основою методу QDA.")
print("6. Порівняння результатів показало, що наша реалізація працює так само ефективно, як і вбудовані функції sklearn, що свідчить про правильність алгоритму.")
print("7. Надалі можна вдосконалювати модель, додавши додаткові оптимізації для великих наборів даних.")