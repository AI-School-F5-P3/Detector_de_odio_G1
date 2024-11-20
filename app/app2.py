
import streamlit as st
from transformers import pipeline
import warnings
import pickle
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Configuración de la página con tema claro
st.set_page_config(
    page_title="Detector de Discurso de Odio",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Aplicar estilo personalizado
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            background-color: #ff4b4b;
            color: white;
        }
        .stButton>button:hover {
            background-color: #ff3333;
        }
    </style>
""", unsafe_allow_html=True)

class ModeloPersonalizado:
    def __init__(self, modelo, vectorizador):
        self.modelo = modelo
        self.vectorizador = vectorizador
    
    def predict(self, texto):
        # Asegurar que el texto sea una lista
        if isinstance(texto, str):
            texto = [texto]
        # Vectorizar el texto
        texto_vectorizado = self.vectorizador.transform(texto)
        # Realizar predicción
        return self.modelo.predict(texto_vectorizado)
    
    def predict_proba(self, texto):
        # Asegurar que el texto sea una lista
        if isinstance(texto, str):
            texto = [texto]
        # Vectorizar el texto
        texto_vectorizado = self.vectorizador.transform(texto)
        # Realizar predicción con probabilidades
        return self.modelo.predict_proba(texto_vectorizado)

def cargar_modelo_desde_cache():
    """Carga el modelo desde caché si existe y no está expirado"""
    cache_file = 'modelo_cache.pkl'
    cache_info_file = 'cache_info.pkl'
    cache_duration = timedelta(days=7)  # El caché expira después de 7 días
    
    try:
        # Verificar si existe información del caché
        if os.path.exists(cache_info_file):
            with open(cache_info_file, 'rb') as f:
                cache_date = pickle.load(f)
            
            # Verificar si el caché ha expirado
            if datetime.now() - cache_date > cache_duration:
                return None
        
        # Cargar modelo desde caché si existe
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Error al cargar el caché: {str(e)}")
    return None

def cargar_modelo_personalizado():
    """Carga el modelo personalizado y el vectorizador desde los archivos"""
    try:
        # Cargar el modelo
        with open('models/modelo_regLog.pkl', 'rb') as f:
            modelo = pickle.load(f)
        
        # Cargar el vectorizador
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizador = pickle.load(f)
        
        # Crear instancia de ModeloPersonalizado que contiene ambos
        return ModeloPersonalizado(modelo, vectorizador)
    except Exception as e:
        st.error(f"Error al cargar el modelo personalizado o vectorizador: {str(e)}")
        return None

def guardar_modelo_en_cache(modelo):
    """Guarda el modelo en caché"""
    try:
        # Guardar el modelo
        with open('modelo_cache.pkl', 'wb') as f:
            pickle.dump(modelo, f)
        
        # Guardar la fecha de caché
        with open('cache_info.pkl', 'wb') as f:
            pickle.dump(datetime.now(), f)
    except Exception as e:
        st.error(f"Error al guardar el caché: {str(e)}")

@st.cache_resource
def cargar_modelo_complex():
    try:
        # Cargar desde complex
        modelo = pipeline(
            "text-classification",
            model="martin-ha/toxic-comment-model",
            framework="pt"
        )
        # Guardar en caché para futuro uso
        guardar_modelo_en_cache(modelo)
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo de complex: {str(e)}")
        return None

def predecir_odio_complex(texto, clasificador):
    try:
        resultado = clasificador(texto)[0]
        es_toxico = resultado['label'] == 'toxic'
        probabilidad = resultado['score']
        return es_toxico, probabilidad
    except Exception as e:
        st.error(f"Error en el procesamiento: {str(e)}")
        return None, None

def predecir_odio_personalizado(texto, modelo):
    try:
        # Obtener probabilidades usando el modelo personalizado
        probabilidades = modelo.predict_proba(texto)
        # Asumiendo que la segunda columna es la probabilidad de la clase positiva (tóxico)
        probabilidad = probabilidades[0][1]
        es_toxico = probabilidad > 0.5
        return es_toxico, probabilidad
    except Exception as e:
        st.error(f"Error en el procesamiento con modelo personalizado: {str(e)}")
        return None, None

def mostrar_tiempo_carga():
    """Muestra el tiempo de carga del modelo"""
    if 'tiempo_inicio' in st.session_state:
        tiempo_fin = datetime.now()
        tiempo_carga = (tiempo_fin - st.session_state.tiempo_inicio).total_seconds()
        st.success(f"✨ Modelo cargado en {tiempo_carga:.2f} segundos")

def main():
    st.title("🛡️ Detector de Contenido Tóxico")
    
    # Registrar tiempo de inicio
    if 'tiempo_inicio' not in st.session_state:
        st.session_state.tiempo_inicio = datetime.now()
    
    st.markdown("""
    Esta aplicación analiza texto en inglés para detectar contenido potencialmente tóxico o dañino.
    
    📝 **Instrucciones:**
    1. Ingresa el texto en inglés que deseas analizar
    2. Selecciona el modelo a utilizar
    3. Espera los resultados del análisis
    """)
    
    # Área de texto para input
    texto_usuario = st.text_area(
        "Ingresa el texto en inglés:",
        height=150,
        placeholder="Escribe o pega aquí el texto a analizar..."
    )
    
    # Columnas para los botones
    col1, col2 = st.columns(2)
    
    with col1:
        analizar_complex = st.button("Usar Modelo complex")
    
    with col2:
        analizar_personalizado = st.button("📊 Usar Modelo Inicial")
    
    if texto_usuario.strip():
        if analizar_complex:
            with st.spinner("Cargando modelo de complex..."):
                modelo = cargar_modelo_desde_cache() or cargar_modelo_complex()
                if modelo:
                    es_toxico, prob = predecir_odio_complex(texto_usuario, modelo)
                    mostrar_resultados(es_toxico, prob)
        
        elif analizar_personalizado:
            with st.spinner("Cargando modelo personalizado..."):
                modelo = cargar_modelo_personalizado()
                if modelo:
                    es_toxico, prob = predecir_odio_personalizado(texto_usuario, modelo)
                    mostrar_resultados(es_toxico, prob)
    else:
        if analizar_complex or analizar_personalizado:
            st.warning("⚠️ Por favor, ingresa texto para analizar.")

def mostrar_resultados(es_toxico, prob):
    if es_toxico is not None:
        st.markdown("---")
        st.markdown("### Resultados del Análisis:")
        
        prob_percentage = prob * 100
        if es_toxico:
            st.error(f"⚠️ Contenido potencialmente tóxico detectado")
            st.progress(prob)
            st.markdown(f"**Probabilidad de contenido tóxico:** {prob_percentage:.1f}%")
        else:
            st.success(f"✅ Contenido no tóxico")
            st.progress(1 - prob)
            st.markdown(f"**Probabilidad de contenido seguro:** {(1-prob)*100:.1f}%")

    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p style='color: gray; font-size: 0.8em'>
                Esta herramienta utiliza IA para analizar texto. 
                Los resultados son orientativos y pueden no ser 100% precisos.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

