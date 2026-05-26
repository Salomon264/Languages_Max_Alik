import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import os

print("🔄 Создание нового векторайзера...")

# Загружаем метаданные
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, 'saved_models')

metadata_file = os.path.join(MODELS_PATH, 'model_metadata_20260525_225551.json')
if not os.path.exists(metadata_file):
    metadata_file = os.path.join(MODELS_PATH, 'model_metadata_20260525_225614.json')

with open(metadata_file, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Получаем параметры из метаданных
ngram_range = tuple(metadata.get('ngram_range', [1, 2]))
max_features = metadata.get('max_features', 5000)

print(f"📊 Параметры: ngram_range={ngram_range}, max_features={max_features}")

# Создаем и обучаем векторайзер
vectorizer = TfidfVectorizer(
    ngram_range=ngram_range,
    max_features=max_features,
    lowercase=True,
    analyzer='word',
    token_pattern=r'(?u)\b\w\w+\b'
)

# Обучаем на примерах текстов (нужно для fit)
sample_texts = [
    "This is a sample text in English language",
    "Это пример текста на русском языке",
    "Ceci est un exemple de texte en français",
    "Dies ist ein Beispieltext auf Deutsch",
    "Este es un texto de ejemplo en español",
    "Questo è un testo di esempio in italiano",
    "这是一个中文示例文本",
    "これは日本語のサンプルテキストです",
    "هذا نص عربي مثال",
    "यह हिंदी में एक नमूना पाठ है"
]

# Обучаем векторайзер
vectorizer.fit(sample_texts)
print(f"✅ Векторайзер обучен на {len(sample_texts)} примерах")
print(f"📊 Размер словаря: {len(vectorizer.get_feature_names_out())}")

# Сохраняем новый векторайзер
new_vectorizer_file = os.path.join(MODELS_PATH, 'vectorizer_working.pkl')
with open(new_vectorizer_file, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"✅ Новый векторайзер сохранен в {new_vectorizer_file}")

# Тестируем
test_text = "Hello world"
test_vector = vectorizer.transform([test_text])
print(f"✅ Тест пройден: текст '{test_text}' векторизован успешно")
print(f"📊 Размер вектора: {test_vector.shape}")