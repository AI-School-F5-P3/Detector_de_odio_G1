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
def cargar_modelo():
    # Intentar cargar desde caché primero
    modelo = cargar_modelo_desde_cache()
    
    if modelo is None:
        try:
            # Si no está en caché, cargar desde HuggingFace
            modelo = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",
                framework="pt"
            )
            # Guardar en caché para futuro uso
            guardar_modelo_en_cache(modelo)
        except Exception as e:
            
            st.error(f"Error al cargar el modelo: {str(e)}")
            return None
    
    return modelo

def predecir_odio(texto, clasificador):
    try:
        resultado = clasificador(texto)[0]
        es_toxico = resultado['label'] == 'toxic'
        probabilidad = resultado['score']
        return es_toxico, probabilidad
    except Exception as e:
        st.error(f"Error en el procesamiento: {str(e)}")
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
    2. Haz clic en 'Analizar Texto'
    3. Espera los resultados del análisis
    """)
    
    # Cargar el modelo y mostrar tiempo
    clasificador = cargar_modelo()
    mostrar_tiempo_carga()
    
    if clasificador is not None:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            texto_usuario = st.text_area(
                "Ingresa el texto en inglés:",
                height=150,
                placeholder="Escribe o pega aquí el texto a analizar..."
            )
        
        with col2:
            st.markdown("<br>" * 4, unsafe_allow_html=True)
            analizar = st.button("🔍 Analizar Texto")
        
        if analizar:
            if texto_usuario.strip():
                with st.spinner("Analizando el texto..."):
                    es_toxico, prob = predecir_odio(texto_usuario, clasificador)
                    
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
            else:
                st.warning("⚠️ Por favor, ingresa texto para analizar.")
    
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