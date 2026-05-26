import pandas as pd
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os

print("🔄 Обучение новой модели...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, 'saved_models')

# Загрузка данных
data_file = os.path.join(BASE_DIR, 'augmented_language_dataset_clean.csv')
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Датасет не найден: {data_file}")

print(f"📂 Загрузка данных из {data_file}")
df = pd.read_csv(data_file)
print(f"✅ Загружено {len(df)} образцов")

# Проверяем колонки
print(f"📊 Колонки: {df.columns.tolist()}")

# Используем правильные названия колонок
text_column = 'clean_text'
language_column = 'Language'

# Проверяем наличие колонок
if text_column not in df.columns:
    raise KeyError(f"Колонка '{text_column}' не найдена. Доступные колонки: {df.columns.tolist()}")
if language_column not in df.columns:
    raise KeyError(f"Колонка '{language_column}' не найдена. Доступные колонки: {df.columns.tolist()}")

# Удаляем строки с NaN
print(f"🔄 Очистка данных...")
before = len(df)
df = df.dropna(subset=[text_column, language_column])
after = len(df)
print(f"✅ Удалено {before - after} строк с пропущенными значениями")

# Удаляем пустые строки
df = df[df[text_column].str.strip() != '']
print(f"✅ После удаления пустых строк: {len(df)} образцов")

print(f"📊 Уникальные языки: {df[language_column].nunique()}")
print(f"📊 Языки: {sorted(df[language_column].unique())}")

# Разделяем на X и y
X = df[text_column].values
y = df[language_column].values

# Векторизация
print("🔄 Векторизация текстов...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    lowercase=True,
    analyzer='word',
    token_pattern=r'(?u)\b\w\w+\b'
)

X_vectorized = vectorizer.fit_transform(X)
print(f"✅ Векторизация завершена. Размер: {X_vectorized.shape}")

# Разделяем на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

# Обучение модели
print("🔄 Обучение модели LinearSVC...")
model = LinearSVC(random_state=42, max_iter=1000, dual='auto')
model.fit(X_train, y_train)
print(f"✅ Модель обучена")

# Оценка
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Точность модели: {accuracy:.4f}")

# Сохранение модели
os.makedirs(MODELS_PATH, exist_ok=True)
model_file = os.path.join(MODELS_PATH, 'model_new.pkl')
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"✅ Модель сохранена в {model_file}")

# Сохранение векторайзера
vectorizer_file = os.path.join(MODELS_PATH, 'vectorizer_new.pkl')
with open(vectorizer_file, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"✅ Векторайзер сохранен в {vectorizer_file}")

# Сохранение метаданных
metadata = {
    'model_type': 'LinearSVC',
    'accuracy': float(accuracy),
    'max_features': 5000,
    'ngram_range': [1, 2],
    'total_languages': len(np.unique(y)),
    'languages': [str(lang) for lang in np.unique(y)],
    'samples_count': len(df),
    'samples_after_cleaning': len(df),
    'timestamp': pd.Timestamp.now().isoformat()
}

metadata_file = os.path.join(MODELS_PATH, 'model_metadata_new.json')
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"✅ Метаданные сохранены")

print(f"\n🎉 Обучение завершено!")
print(f"📊 Всего языков: {len(metadata['languages'])}")
print(f"📊 Список языков: {metadata['languages']}")
print(f"📊 Точность: {accuracy:.2%}")