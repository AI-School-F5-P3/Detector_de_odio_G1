import streamlit as st
from transformers import pipeline
import warnings
import pickle
import os
from datetime import datetime, timedelta
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re
warnings.filterwarnings('ignore')


import sys
import traceback

def verificar_dependencias():
    """Verificar si todas las dependencias están instaladas correctamente"""
    dependencias = [
        'transformers', 
        'torch', 
        'pandas', 
        'googleapiclient'
    ]
    
    for dep in dependencias:
        try:
            __import__(dep)
        except ImportError:
            st.error(f"Dependencia faltante: {dep}")
            st.error("Instala las dependencias con: pip install transformers torch pandas google-api-python-client")
            return False
    return True
print(verificar_dependencias())

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Comentarios de YouTube",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
    <style>
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
        }
        .stButton>button:hover {
            background-color: #ff3333;
        }
        .toxic-comment {
            background-color: #ffebee;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .safe-comment {
            background-color: #e8f5e9;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Función para cargar el modelo desde caché
def cargar_modelo_desde_cache():
    cache_file = 'modelo_cache.pkl'
    cache_info_file = 'cache_info.pkl'
    cache_duration = timedelta(days=7)
    
    try:
        if os.path.exists(cache_info_file):
            with open(cache_info_file, 'rb') as f:
                cache_date = pickle.load(f)
            if datetime.now() - cache_date > cache_duration:
                return None
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Error al cargar el caché: {str(e)}")
    return None

def guardar_modelo_en_cache(modelo):
    try:
        with open('modelo_cache.pkl', 'wb') as f:
            pickle.dump(modelo, f)
        with open('cache_info.pkl', 'wb') as f:
            pickle.dump(datetime.now(), f)
    except Exception as e:
        st.error(f"Error al guardar el caché: {str(e)}")

@st.cache_resource
def cargar_modelo():
    modelo = cargar_modelo_desde_cache()
    if modelo is None:
        try:
            modelo = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",
                framework="pt"
            )
            guardar_modelo_en_cache(modelo)
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            return None
    return modelo


def conectar_youtube_api(api_key):
    """
    Crear servicio de YouTube API
    
    Args:
        api_key (str): Tu clave de YouTube API
    
    Returns:
        Objeto de servicio de YouTube para realizar consultas
    """
    api_key=' __'
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        return youtube
    except Exception as e:
        st.error(f"Error al conectar con YouTube API: {e}")
        return None


def extraer_video_id(url):
    """Extrae el ID del video de una URL de YouTube"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def obtener_comentarios_youtube(api_key, video_id, max_results=100):
    """Obtiene comentarios de un video de YouTube"""
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    try:
        comentarios = []
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=max_results,
            textFormat='plainText'
        )
        
        while request:
            response = request.execute()
            
            for item in response['items']:
                comentario = {
                    'texto': item['snippet']['topLevelComment']['snippet']['textDisplay'],
                    'autor': item['snippet']['topLevelComment']['snippet']['authorDisplayName'],
                    'fecha': item['snippet']['topLevelComment']['snippet']['publishedAt'],
                    'likes': item['snippet']['topLevelComment']['snippet']['likeCount']
                }
                comentarios.append(comentario)
            
            request = youtube.commentThreads().list_next(request, response)
            
            if len(comentarios) >= max_results:
                break
                
        return pd.DataFrame(comentarios)
    
    except HttpError as e:
        st.error(f"Error al obtener comentarios: {str(e)}")
        return None

def analizar_toxicidad(texto, clasificador):
    """Analiza la toxicidad de un texto"""
    try:
        resultado = clasificador(texto)[0]
        es_toxico = resultado['label'] == 'toxic'
        probabilidad = resultado['score']
        return es_toxico, probabilidad
    except Exception as e:
        return None, None

def mostrar_estadisticas(df):
    """Muestra estadísticas de los comentarios analizados"""
    st.markdown("### 📊 Estadísticas del Análisis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Comentarios", len(df))
    with col2:
        toxic_percentage = (df['es_toxico'].sum() / len(df)) * 100
        st.metric("Comentarios Tóxicos", f"{toxic_percentage:.1f}%")
    with col3:
        avg_toxicity = df['probabilidad_toxicidad'].mean() * 100
        st.metric("Toxicidad Promedio", f"{avg_toxicity:.1f}%")
    with col4:
        st.metric("Likes Totales", df['likes'].sum())

def main():
    st.title("🎥 Analizador de Comentarios de YouTube")
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        api_key = st.text_input("API Key de YouTube", type="password")
        max_comments = st.slider("Número máximo de comentarios a analizar", 10, 100, 50)
        st.markdown("""
        ### 📝 Instrucciones:
        1. Ingresa tu API Key de YouTube
        2. Pega la URL del video
        3. Ajusta el número de comentarios
        4. Haz clic en 'Analizar Comentarios'
        """)
    
    # Área principal
    url_video = st.text_input("URL del Video de YouTube:", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("🔍 Analizar Comentarios") and api_key and url_video:
        video_id = extraer_video_id(url_video)
        
        if not video_id:
            st.error("URL de video no válida")
            return
        
        with st.spinner("Cargando el modelo de análisis..."):
            clasificador = cargar_modelo()
        
        if clasificador:
            with st.spinner("Obteniendo comentarios..."):
                df_comentarios = obtener_comentarios_youtube(api_key, video_id, max_comments)
                
                if df_comentarios is not None and not df_comentarios.empty:
                    # Analizar comentarios
                    resultados = []
                    for _, row in df_comentarios.iterrows():
                        es_toxico, prob = analizar_toxicidad(row['texto'], clasificador)
                        resultados.append({
                            'es_toxico': es_toxico,
                            'probabilidad_toxicidad': prob
                        })
                    
                    # Agregar resultados al DataFrame
                    resultados_df = pd.DataFrame(resultados)
                    df_comentarios = pd.concat([df_comentarios, resultados_df], axis=1)
                    
                    # Mostrar estadísticas
                    mostrar_estadisticas(df_comentarios)
                    
                    # Mostrar resultados detallados
                    st.markdown("### 💬 Análisis de Comentarios")
                    
                    # Filtros
                    col1, col2 = st.columns(2)
                    with col1:
                        mostrar_toxicos = st.checkbox("Mostrar solo comentarios tóxicos", False)
                    with col2:
                        orden = st.selectbox("Ordenar por:", ["Toxicidad", "Likes", "Fecha"])
                    
                    # Aplicar filtros
                    if mostrar_toxicos:
                        df_filtrado = df_comentarios[df_comentarios['es_toxico']]
                    else:
                        df_filtrado = df_comentarios
                    
                    if orden == "Toxicidad":
                        df_filtrado = df_filtrado.sort_values('probabilidad_toxicidad', ascending=False)
                    elif orden == "Likes":
                        df_filtrado = df_filtrado.sort_values('likes', ascending=False)
                    else:  # Fecha
                        df_filtrado = df_filtrado.sort_values('fecha', ascending=False)
                    
                    # Mostrar comentarios
                    for _, row in df_filtrado.iterrows():
                        clase_css = "toxic-comment" if row['es_toxico'] else "safe-comment"
                        st.markdown(f"""
                            <div class="{clase_css}">
                                <p><strong>👤 {row['autor']}</strong> • ❤️ {row['likes']} likes</p>
                                <p>{row['texto']}</p>
                                <p><em>Probabilidad de toxicidad: {row['probabilidad_toxicidad']*100:.1f}%</em></p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Opción para descargar resultados
                    csv = df_comentarios.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Descargar Resultados (CSV)",
                        csv,
                        "youtube_comments_analysis.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.warning("No se pudieron obtener comentarios del video")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p style='color: gray; font-size: 0.8em'>
                Esta herramienta utiliza la API de YouTube y modelos de IA para analizar comentarios.
                Los resultados son orientativos y pueden no ser 100% precisos.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
