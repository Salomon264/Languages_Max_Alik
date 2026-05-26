from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
from typing import Dict, List, Optional
import json
import os

app = FastAPI(
    title="Language Detection API",
    description="API для определения языка текста",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔄 Загрузка моделей...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, 'saved_models')

# Загрузка новой модели
model_file = os.path.join(MODELS_PATH, 'model_new.pkl')
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Модель не найдена. Сначала запусти train_new_model.py")

print(f"📂 Загрузка модели из {model_file}")
with open(model_file, 'rb') as f:
    model = pickle.load(f)
print(f"✅ Модель загружена. Тип: {type(model).__name__}")

# Загрузка векторайзера
vectorizer_file = os.path.join(MODELS_PATH, 'vectorizer_new.pkl')
if not os.path.exists(vectorizer_file):
    raise FileNotFoundError(f"Векторайзер не найден. Сначала запусти train_new_model.py")

print(f"📂 Загрузка векторайзера из {vectorizer_file}")
with open(vectorizer_file, 'rb') as f:
    vectorizer = pickle.load(f)
print(f"✅ Векторайзер загружен")

# Загрузка метаданных
metadata_file = os.path.join(MODELS_PATH, 'model_metadata_new.json')
if os.path.exists(metadata_file):
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"✅ Метаданные загружены")
else:
    metadata = {}

# Получаем список языков
if hasattr(model, 'classes_'):
    classes = [str(c) for c in model.classes_]
else:
    classes = metadata.get('languages', [])

print(f"🎯 Загружено {len(classes)} языков")
print(f"📊 Первые 10: {classes[:10] if len(classes) > 10 else classes}")

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="Текст для определения языка")

class Top3Item(BaseModel):
    language: str
    probability: float

class PredictionResponse(BaseModel):
    language: str
    confidence: float
    probabilities: Dict[str, float]
    top_3: List[Top3Item]

class LanguagesResponse(BaseModel):
    languages: List[str]
    count: int

class StatsResponse(BaseModel):
    model_type: str
    accuracy: float
    features_count: int
    ngram_range: List[int]
    total_languages: int
    samples_count: int
    timestamp: Optional[str] = None

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "Language Detection API",
        "version": "1.0.0",
        "available_languages": len(classes),
        "model_type": type(model).__name__,
        "accuracy": f"{metadata.get('accuracy', 0) * 100:.1f}%"
    }

@app.get("/languages", response_model=LanguagesResponse)
def get_languages():
    return {"languages": classes, "count": len(classes)}

@app.get("/stats", response_model=StatsResponse)
def get_model_stats():
    return {
        "model_type": metadata.get('model_type', type(model).__name__),
        "accuracy": metadata.get('accuracy', 0.85),
        "features_count": metadata.get('max_features', 5000),
        "ngram_range": metadata.get('ngram_range', [1, 2]),
        "total_languages": len(classes),
        "samples_count": metadata.get('samples_count', 0),
        "timestamp": metadata.get('timestamp', None)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_language(item: TextInput):
    if not item.text or len(item.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Текст должен содержать минимум 3 символа")
    
    # Векторизация
    text_vectorized = vectorizer.transform([item.text])
    
    # Предсказание
    prediction = str(model.predict(text_vectorized)[0])
    
    # Получение вероятностей через decision_function
    if hasattr(model, 'decision_function'):
        decisions = model.decision_function(text_vectorized)
        if len(decisions.shape) == 1:
            decisions = decisions.reshape(1, -1)
        decisions = decisions[0]
        exp = np.exp(decisions - np.max(decisions))
        probabilities = exp / exp.sum()
    else:
        probabilities = np.ones(len(classes)) / len(classes)
    
    # Создаем словарь вероятностей
    prob_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    
    # Топ-3
    top_3_items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    top_3_list = [Top3Item(language=lang, probability=prob) for lang, prob in top_3_items]
    
    # Уверенность
    confidence = max(prob_dict.values())
    
    return PredictionResponse(
        language=prediction,
        confidence=round(confidence, 4),
        probabilities=prob_dict,
        top_3=top_3_list
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Запуск Language Detection API...")
    print(f"📍 http://127.0.0.1:8000")
    print(f"📖 http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)