# -*- coding: utf-8 -*-
"""hw_03_svertoka_viktor.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XbKTDD5yoyKM7c24hjqLAgfo0ZGk9Y4W
"""

# Імпорт необхідних бібліотек
import numpy as np
import pandas as pd
import pickle  # Для роботи з файлом у форматі .p

# Завантаження NLP-моделі
# Замість 'word_embeddings_file' вкажіть шлях до файлу з моделлю
word_embeddings_file = 'word_embeddings_subset.p'

# Завантаження моделі з використанням pickle
with open(word_embeddings_file, 'rb') as f:
    model = pickle.load(f)

# Отримання слів та їхніх тривимірних векторів
words = list(model.keys())
vectors = np.array([model[word][:3] for word in words])  # Беремо лише перші 3 координати

# Створення DataFrame
df = pd.DataFrame(vectors, index=words, columns=['x', 'y', 'z'])
df.reset_index(inplace=True)
df.rename(columns={'index': 'word'}, inplace=True)

# Перевірка результату
print(f"{'-' * 20} 1 {'-' * 20}")

print("Слова в наборі даних:\n", df['word'].head())

print(f"{'-' * 20} 2 {'-' * 20}")

print(df.head())



# Функція для пошуку найближчого слова
def find_closest_word(vector, df):
    """
    Знаходить слово, найближче до заданого вектора.
    """
    # Розрахунок косинусної подібності
    similarities = df[['x', 'y', 'z']].apply(
        lambda row: np.dot(vector, row) / (np.linalg.norm(vector) * np.linalg.norm(row)),
        axis=1
    )
    # Найближче слово
    closest_idx = similarities.idxmax()
    return df.loc[closest_idx, 'word']

# Приклад використання
test_vector = np.array([0.1, 0.2, 0.3])

print(f"{'-' * 20} 3 {'-' * 20}")

print("Найближче слово:", find_closest_word(test_vector, df))

# Функція для пошуку ортогонального слова
def find_orthogonal_word(word1, word2, df):
    """
    Знаходить найближче слово до векторного добутку двох слів.
    """
    vec1 = df.loc[df['word'] == word1, ['x', 'y', 'z']].values[0]
    vec2 = df.loc[df['word'] == word2, ['x', 'y', 'z']].values[0]
    cross_product = np.cross(vec1, vec2)
    return find_closest_word(cross_product, df)

# Приклад використання
word1, word2 = "king", "queen"

print(f"{'-' * 20} 4 {'-' * 20}")

print(f"Ортогональне слово для {word1} і {word2}: {find_orthogonal_word(word1, word2, df)}")

# Функція для обчислення кута між векторами двох слів
def calculate_angle(word1, word2, df):
    """
    Обчислює кут між векторами двох слів у градусах.
    """
    if word1 not in df['word'].values:
        return f"Слово {word1} не знайдено у наборі даних."
    if word2 not in df['word'].values:
        return f"Слово {word2} не знайдено у наборі даних."

    vec1 = df.loc[df['word'] == word1, ['x', 'y', 'z']].values[0]
    vec2 = df.loc[df['word'] == word2, ['x', 'y', 'z']].values[0]
    cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cosine_similarity, -1.0, 1.0))
    return np.degrees(angle)


# Приклад використання
word1, word2 = "city", "country"

print(f"{'-' * 20} 5 {'-' * 20}")

print(f"Кут між {word1} і {word2}: {calculate_angle(word1, word2, df)} градусів")

print(f"{'-' * 43}")

print("""
Висновки:

Реалізовані функції демонструють роботу з векторами слів,
включаючи знаходження найближчих слів,
обчислення ортогональних слів та кутів між словами.

Отримані результати допомагають інтерпретувати семантичні зв’язки
між словами у тривимірному просторі.
""")