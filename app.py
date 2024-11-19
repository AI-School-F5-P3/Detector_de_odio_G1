import streamlit as st
import pickle


# Cargar el modelo y el vectorizador entrenados

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    
with open('modelo_regLog.pkl', 'rb') as f:
    model = pickle.load(f)
    


# Función para clasificar un mensaje
def classify_message(text):
    # Transformar el texto en un vector de características
    X = vectorizer.transform([text])
    # Realizar la predicción utilizando el modelo entrenado
    y_pred = model.predict(X)[0]
    # Devolver el resultado de la clasificación
    return bool(y_pred)

# Crear la aplicación Streamlit
st.title("Verificador de Mensajes Odiosos")
st.write("Ingresa un mensaje y te diré si es considerado odioso o no.")

# Obtener el mensaje del usuario
message = st.text_area("Mensaje a verificar:", height=100)

# Clasificar el mensaje y mostrar el resultado
if st.button("Verificar mensaje"):
    if classify_message(message):
        st.write("El mensaje es considerado como odioso.")
    else:
        st.write("El mensaje no es considerado como odioso.")