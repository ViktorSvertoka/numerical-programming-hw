# -*- coding: utf-8 -*-
"""hw_04_svertoka_viktor.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XBOJBTPdUrZN95hNqCqmz_z29-2rMO9G
"""

# Імпорт необхідних бібліотек
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Завантажте та ознайомтесь з даними
data = load_breast_cancer()

# Опис набору даних
print("Опис набору даних:\n", data.DESCR)

# 2. Створіть DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Переглядаємо перші кілька рядків
print("\nПерші 5 рядків DataFrame:\n", df.head())

# 3. Виведіть інформацію про дані
print("\nІнформація про DataFrame:")
df_info = df.info()

# 4. Виведіть описові статистики
print("\nОписові статистики для числових стовпців:")
df_desc = df.describe()

# 5. Стандартизуйте дані
scaler = StandardScaler()
df_scaled = df.drop('target', axis=1)  # Вилучаємо цільову змінну
df_scaled = scaler.fit_transform(df_scaled)

# Переводимо назад у DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=data.feature_names)
print("\nПерші 5 рядків стандартизованих даних:")
print(df_scaled.head())

# 6. Побудуйте точкові діаграми
print("\nПобудова точкових діаграм між усіма стовпцями:")
sns.pairplot(df, hue="target")
plt.show()

# 7. Обчисліть матриці відстаней для різних метрик
metrics = ['cityblock', 'cosine', 'euclidean']
distance_matrices = {}

for metric in metrics:
    distance_matrices[metric] = cdist(df_scaled, df_scaled, metric=metric)

# Виведення матриці відстаней для метрики "euclidean"
print("\nМатриця відстаней для метрики 'euclidean':")
print(distance_matrices['euclidean'])


# 8. Візуалізуйте отримані матриці
print("\nВізуалізація матриці відстаней для метрики 'euclidean':")
sns.heatmap(distance_matrices['euclidean'], cmap="YlGnBu", xticklabels=False, yticklabels=False)
plt.title("Матриця відстаней (Euclidean)")
plt.show()

# 9. Зробіть висновок
print("\nВисновки:")
print("""
1. Завантажено набір даних Breast Cancer та переглянуто його опис.
2. Створено DataFrame для зручної роботи з даними.
3. Отримано основну інформацію про типи даних та наявність пропусків.
4. Виведено описові статистики, що дозволяють оцінити розподіл числових значень.
5. Стандартизовано дані для коректного обчислення відстаней.
6. Побудовано точкові діаграми для візуалізації взаємозв'язків між ознаками.
7. Обчислено матриці відстаней для різних метрик, таких як cityblock, cosine, euclidean, l1, manhattan.
8. Візуалізовано матрицю відстаней для метрики "euclidean" за допомогою теплової карти.
9. На основі отриманих результатів можна зробити висновки про схожість та відмінність між різними об'єктами набору даних.
""")