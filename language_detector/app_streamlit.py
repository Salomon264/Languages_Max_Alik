"""
Приложение с графическим интерфейсом для определения языка текста
Пункт 8: Разработка приложения с графическим интерфейсом
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time
import json

# ==================== НАСТРОЙКА СТРАНИЦЫ ====================
st.set_page_config(
    page_title="Language Detector",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"

# ==================== ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДАННЫХ ====================
@st.cache_data
def load_dataset():
    """Загрузка датасета"""
    try:
        df = pd.read_csv('augmented_language_dataset_clean.csv')
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки датасета: {e}")
        return None

@st.cache_data
def get_api_info():
    """Получение информации об API"""
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        if response.status_code == 200:
            return response.json()
    except:
        return None

@st.cache_data
def get_languages():
    """Получение списка языков из API"""
    try:
        response = requests.get(f"{API_URL}/languages", timeout=3)
        if response.status_code == 200:
            return response.json().get("languages", [])
    except:
        return []
    return []

@st.cache_data
def get_model_stats():
    """Получение статистики модели"""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=3)
        if response.status_code == 200:
            return response.json()
    except:
        return {}
    return {}

def predict_language(text):
    """Отправка текста на API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Ошибка {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "API сервер не запущен. Запустите: uvicorn api:app --reload"}
    except Exception as e:
        return {"error": f"Ошибка: {str(e)}"}

# ==================== БОКОВАЯ ПАНЕЛЬ ====================
with st.sidebar:
    st.title("🌐 Language Detector")
    st.markdown("---")
    
    # Статус API
    st.subheader("🔌 Статус API")
    api_info = get_api_info()
    if api_info:
        st.success("✅ API работает")
        st.caption(f"Версия: {api_info.get('version', '1.0.0')}")
    else:
        st.error("❌ API не отвечает")
        st.caption("Запустите: uvicorn api:app --reload")
    
    st.markdown("---")
    
    # Статистика модели
    st.subheader("📊 Статистика модели")
    stats = get_model_stats()
    if stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Точность", f"{stats.get('accuracy', 0)*100:.1f}%")
            st.metric("🌍 Языков", stats.get('total_languages', 0))
        with col2:
            st.metric("🔤 Признаков", f"{stats.get('features_count', 0):,}")
            st.metric("🏷️ N-граммы", f"1-{stats.get('ngram_range', [1,2])[1]}")
    
    st.markdown("---")
    st.caption("FastAPI + Streamlit | ML Language Detection")

# ==================== ОСНОВНАЯ ОБЛАСТЬ ====================
st.title("🌐 Определение языка текста")
st.markdown("Введите текст на любом языке, и модель определит, на каком языке он написан.")

# ==================== ВКЛАДКИ ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Определение языка",
    "📊 Статистика датасета",
    "📈 Визуализации",
    "🆘 Справка"
])

# ==================== ВКЛАДКА 1: ОПРЕДЕЛЕНИЕ ЯЗЫКА ====================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_text = st.text_area(
            "✍️ Введите текст для анализа:",
            height=200,
            placeholder="Например: Hello, how are you today? 或 今天天气真好！"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_btn = st.button("🔍 Определить язык", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Очистить", use_container_width=True)
            if clear_btn:
                input_text = ""
                st.rerun()
        
        if predict_btn and input_text.strip():
            with st.spinner("Анализируем текст..."):
                result = predict_language(input_text)
            
            if "error" in result:
                st.error(result["error"])
            else:
                # Результат
                st.success(f"**Определённый язык:** {result['language']}")
                
                # Прогресс-бар уверенности
                confidence = result.get('confidence', 0)
                st.progress(confidence, text=f"Уверенность: {confidence*100:.2f}%")
                
                # Топ-3 языка
                st.subheader("🏆 Топ-3 языка")
                top_3 = result.get('top_3', [])
                for item in top_3:
                    st.write(f"- **{item['language']}**: {item['probability']*100:.2f}%")
                
                # Детальные вероятности
                with st.expander("📊 Подробные вероятности по всем языкам"):
                    probs = result.get('probabilities', {})
                    if probs:
                        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                        for lang, prob in sorted_probs:
                            st.write(f"{lang}: {prob*100:.2f}%")
    
    with col2:
        st.markdown("### 📝 Примеры текстов")
        examples = {
            "🇬🇧 English": "The weather is beautiful today! I love spending time outdoors.",
            "🇷🇺 Russian": "Сегодня прекрасная погода! Я люблю проводить время на свежем воздухе.",
            "🇫🇷 French": "Le temps est magnifique aujourd'hui! J'adore passer du temps dehors.",
            "🇩🇪 German": "Das Wetter ist heute wunderschön! Ich verbringe gerne Zeit im Freien.",
            "🇪🇸 Spanish": "¡El clima está hermoso hoy! Me encanta pasar tiempo al aire libre.",
            "🇮🇳 Hindi": "आज मौसम बहुत सुंदर है! मुझे बाहर समय बिताना पसंद है।"
        }
        
        for lang, text in examples.items():
            if st.button(f"📋 {lang}", key=lang):
                st.session_state.example_text = text
                st.rerun()
        
        if "example_text" in st.session_state:
            st.info(f"Пример загружен:\n{st.session_state.example_text[:100]}...")

# ==================== ВКЛАДКА 2: СТАТИСТИКА ДАТАСЕТА ====================
with tab2:
    st.header("📊 Статистика датасета")
    
    df = load_dataset()
    
    if df is not None:
        # Основные метрики
        col1, col2, col3, col4 = st.columns(4)
        
        df['text_length'] = df['clean_text'].astype(str).str.len()
        
        with col1:
            st.metric("📄 Всего текстов", f"{len(df):,}")
        with col2:
            st.metric("🌍 Количество языков", df['Language'].nunique())
        with col3:
            st.metric("📏 Средняя длина", f"{df['text_length'].mean():.0f} симв.")
        with col4:
            st.metric("📐 Медианная длина", f"{df['text_length'].median():.0f} симв.")
        
        st.markdown("---")
        
        # Распределение по языкам
        st.subheader("📊 Распределение текстов по языкам")
        lang_counts = df['Language'].value_counts().reset_index()
        lang_counts.columns = ['Language', 'Count']
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig = px.bar(
                lang_counts.head(15),
                x='Language',
                y='Count',
                title="Количество текстов по языкам (топ-15)",
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            fig_pie = px.pie(
                lang_counts.head(10),
                values='Count',
                names='Language',
                title="Распределение языков (топ-10)"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Статистика по языкам
        st.subheader("📋 Подробная статистика по языкам")
        
        lang_stats = df.groupby('Language').agg({
            'clean_text': 'count',
            'text_length': ['mean', 'min', 'max', 'std']
        }).round(2)
        lang_stats.columns = ['Количество', 'Ср. длина', 'Мин. длина', 'Макс. длина', 'Стд. отклонение']
        lang_stats = lang_stats.sort_values('Количество', ascending=False)
        
        st.dataframe(lang_stats, use_container_width=True, height=400)
        
    else:
        st.error("Не удалось загрузить датасет. Проверьте файл 'augmented_language_dataset_clean.csv'")

# ==================== ВКЛАДКА 3: ВИЗУАЛИЗАЦИИ ====================
with tab3:
    st.header("📈 Визуализации данных")
    
    df = load_dataset()
    
    if df is not None:
        df['text_length'] = df['clean_text'].astype(str).str.len()
        lang_counts = df['Language'].value_counts()
        
        # Диаграмма 1: Treemap
        st.subheader("🗺️ Древо языков")
        fig_treemap = px.treemap(
            names=lang_counts.index,
            parents=[""] * len(lang_counts),
            values=lang_counts.values,
            title="Иерархия языков",
            color=lang_counts.values,
            color_continuous_scale='Rainbow',
            height=500
        )
        st.plotly_chart(fig_treemap, use_container_width=True)
        
        # Диаграмма 2: Sunburst
        st.subheader("☀️ Солнечная диаграмма")
        fig_sunburst = px.sunburst(
            names=lang_counts.index,
            parents=[""] * len(lang_counts),
            values=lang_counts.values,
            title="Распределение языков",
            color=lang_counts.values,
            color_continuous_scale='Viridis',
            height=500
        )
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
        # Диаграмма 3: Гистограмма длин
        st.subheader("📏 Распределение длин текстов")
        fig_hist = px.histogram(
            df,
            x='text_length',
            nbins=50,
            title="Гистограмма длины текстов",
            labels={'text_length': 'Длина текста (символы)', 'count': 'Количество'},
            color_discrete_sequence=['skyblue']
        )
        fig_hist.update_layout(height=450)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Диаграмма 4: Box-plot по языкам
        st.subheader("📦 Распределение длин по языкам")
        top_langs = lang_counts.head(10).index.tolist()
        df_top = df[df['Language'].isin(top_langs)]
        
        fig_box = px.box(
            df_top,
            x='Language',
            y='text_length',
            title="Box-plot длины текста по языкам (топ-10)",
            color='Language',
            height=500
        )
        fig_box.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Диаграмма 5: Радар по частотам
        st.subheader("📡 Радар частотности языков")
        top_8 = lang_counts.head(8)
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=top_8.values,
            theta=top_8.index,
            fill='toself',
            marker=dict(color='blue', size=8)
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Радар частотности языков (топ-8)",
            height=500
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    else:
        st.error("Не удалось загрузить датасет для визуализации")