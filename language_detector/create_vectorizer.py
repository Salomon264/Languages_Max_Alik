import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import os

print("🔄 Создание нового векторайзера...")

# Загружаем метаданные
with open('model_metadata_20260525_225943.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Получаем параметры из метаданных
ngram_range = tuple(metadata.get('ngram_range', [1, 2]))
max_features = metadata.get('max_features', 5000)

print(f"📊 Параметры: ngram_range={ngram_range}, max_features={max_features}")

# Создаем новый векторайзер с теми же параметрами
vectorizer = TfidfVectorizer(
    ngram_range=ngram_range,
    max_features=max_features,
    lowercase=True,
    analyzer='word',
    token_pattern=r'(?u)\b\w\w+\b'
)

# Обучаем векторайзер на небольшом наборе текстов (нужно для fit)
# Берем примеры текстов на разных языках
sample_texts = [
    "This is a sample text in English language for training the vectorizer",
    "Это пример текста на русском языке для обучения векторайзера",
    "Ceci est un exemple de texte en français pour entraîner le vectoriseur",
    "Dies ist ein Beispieltext auf Deutsch zum Trainieren des Vektorisierers",
    "Este es un texto de ejemplo en español para entrenar el vectorizador",
    "Questo è un testo di esempio in italiano per addestrare il vettorizzatore",
    "这是一个中文示例文本用于训练向量化器",
    "これはベクトライザを訓練するための日本語のサンプルテキストです",
    "هذا نص عربي لتدريب جهاز توجيه النص",
    "यह वेक्टराइज़र को प्रशिक्षित करने के लिए हिंदी में एक नमूना पाठ है"
]

# Обучаем векторайзер
vectorizer.fit(sample_texts)
print("✅ Векторайзер обучен на примерах")

# Сохраняем новый векторайзер
new_vectorizer_file = 'vectorizer_new.pkl'
with open(new_vectorizer_file, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"✅ Новый векторайзер сохранен в {new_vectorizer_file}")
print(f"📊 Размер словаря: {len(vectorizer.get_feature_names_out())}")

# Тестируем
test_text = "Hello world"
test_vector = vectorizer.transform([test_text])
print(f"✅ Тест пройден: текст '{test_text}' векторизован успешно")
print(f"📊 Размер вектора: {test_vector.shape}")