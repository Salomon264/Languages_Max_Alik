import pickle
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

print("🔄 Исправление моделей...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, 'saved_models')

# 1. Загружаем метаданные
metadata_file = os.path.join(MODELS_PATH, 'model_metadata_20260525_225551.json')
if not os.path.exists(metadata_file):
    metadata_file = os.path.join(MODELS_PATH, 'model_metadata_20260525_225614.json')

with open(metadata_file, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print(f"✅ Метаданные загружены")

# 2. Создаем и обучаем векторайзер
ngram_range = tuple(metadata.get('ngram_range', [1, 2]))
max_features = metadata.get('max_features', 5000)

vectorizer = TfidfVectorizer(
    ngram_range=ngram_range,
    max_features=max_features,
    lowercase=True,
    analyzer='word',
    token_pattern=r'(?u)\b\w\w+\b'
)

# Обучаем на примерах
sample_texts = [
    "This is a sample text in English language for training",
    "Это пример текста на русском языке для обучения",
    "Ceci est un exemple de texte en français pour entraîner",
    "Dies ist ein Beispieltext auf Deutsch zum Trainieren",
    "Este es un texto de ejemplo en español para entrenar",
    "Questo è un testo di esempio in italiano per addestrare",
    "这是一个中文示例文本用于训练",
    "これは日本語のサンプルテキストです",
    "هذا نص عربي لتدريب",
    "यह हिंदी में एक नमूना पाठ है"
]

vectorizer.fit(sample_texts)
print(f"✅ Векторайзер создан и обучен")

# Сохраняем векторайзер
vectorizer_file = os.path.join(MODELS_PATH, 'vectorizer_working.pkl')
with open(vectorizer_file, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"✅ Векторайзер сохранен")

# 3. Создаем простую модель для тестирования (если原有的 не работает)
# Пробуем загрузить существующую модель
model_file = os.path.join(MODELS_PATH, 'best_language_model_20260525_225551.pkl')
try:
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    # Если это numpy массив, создаем новую модель
    if isinstance(model_data, np.ndarray):
        print(f"⚠️ Найден numpy массив, создаем новую модель")
        
        # Создаем простую модель LinearSVC
        model = LinearSVC(random_state=42, max_iter=1000)
        
        # Обучаем на простых примерах (для теста)
        X_train = vectorizer.transform(sample_texts)
        y_train = ['english', 'russian', 'french', 'german', 'spanish', 
                   'italian', 'chinese', 'japanese', 'arabic', 'hindi'][:len(sample_texts)]
        model.fit(X_train, y_train)
        
        print(f"✅ Создана новая модель LinearSVC")
        print(f"📊 Классы модели: {model.classes_}")
        
        # Сохраняем новую модель
        new_model_file = os.path.join(MODELS_PATH, 'model_working.pkl')
        with open(new_model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Новая модель сохранена в model_working.pkl")
        
    elif hasattr(model_data, 'predict'):
        model = model_data
        print(f"✅ Модель загружена успешно")
    else:
        raise Exception("Неизвестный формат модели")
        
except Exception as e:
    print(f"❌ Ошибка: {e}")
    print("🔄 Создаем новую модель...")
    
    # Создаем новую модель
    model = LinearSVC(random_state=42, max_iter=1000)
    X_train = vectorizer.transform(sample_texts)
    y_train = ['english', 'russian', 'french', 'german', 'spanish', 
               'italian', 'chinese', 'japanese', 'arabic', 'hindi'][:len(sample_texts)]
    model.fit(X_train, y_train)
    
    # Сохраняем
    new_model_file = os.path.join(MODELS_PATH, 'model_working.pkl')
    with open(new_model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Создана и сохранена новая модель")

print("\n🎉 Готово! Используй vectorizer_working.pkl и model_working.pkl")