import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configuración de la página
st.set_page_config(page_title="Detector de Discurso de Odio", page_icon="🔍")

@st.cache_resource
def cargar_modelo():
    # Usamos un modelo pre-entrenado para clasificación de discurso de odio
    modelo_nombre = "facebook/roberta-hate-speech-dynabench-r4-target"
    tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
    modelo = AutoModelForSequenceClassification.from_pretrained(modelo_nombre)
    return tokenizer, modelo

def predecir_odio(texto, tokenizer, modelo):
    # Preparar el texto para el modelo
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Realizar la predicción
    with torch.no_grad():
        outputs = modelo(**inputs)
        predicciones = torch.softmax(outputs.logits, dim=1)
        
    # Obtener la clase predicha y la probabilidad
    clase_predicha = torch.argmax(predicciones).item()
    probabilidad = predicciones[0][clase_predicha].item()
    
    return clase_predicha, probabilidad

def main():
    st.title("📝 Detector de Discurso de Odio")
    st.write("Esta aplicación analiza texto en inglés para detectar posible discurso de odio.")
    
    # Cargar el modelo
    tokenizer, modelo = cargar_modelo()
    
    # Área de texto para input
    texto_usuario = st.text_area("Ingresa el texto en inglés a analizar:", height=150)
    
    if st.button("Analizar"):
        if texto_usuario.strip():
            with st.spinner("Analizando el texto..."):
                clase, prob = predecir_odio(texto_usuario, tokenizer, modelo)
                
                # Mostrar resultados
                st.write("---")
                st.write("### Resultados:")
                
                if clase == 1:
                    st.error(f"⚠️ Se detectó discurso de odio (Probabilidad: {prob:.2%})")
                else:
                    st.success(f"✅ No se detectó discurso de odio (Probabilidad: {prob:.2%})")
        else:
            st.warning("Por favor, ingresa algún texto para analizar.")


if __name__ == "__main__":
    main()