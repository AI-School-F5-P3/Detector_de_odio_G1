
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Cargar el dataset
df = pd.read_csv('C:/4_F5/019_NPL/Detector_de_odio_G1/youtoxic_english_1000.csv', sep=',')


#from pandas_profiling import ProfileReport
#from ydata_profiling import ProfileReport #para Python 3.12


# Generar un reporte

""" 
profile_hate= ProfileReport(df,title="Reporte Exploratorio")
ruta="C:/4_F5/019_NPL/Detector_de_odio_G1/"
profile_hate.to_file(ruta+"reporte_hate.html") 
"""


# Obtener información general sobre el conjunto de datos
print(df.info())

print(df.head())

# Verificar la distribución de las etiquetas
print(df.sum())


# Analizar valores faltantes
print("Cantidad de valores faltantes por columna:")
print(df.isnull().sum())


# Calcular el porcentaje de valores faltantes por columna
print("\nPorcentaje de valores faltantes por columna:")
print((df.isnull().sum() / len(df) * 100).round(2))



# Visualizar valores faltantes
#using a Jupyter Notebook to enable inline plotting needs this--> %matplotlib inline 

#get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Mapa de calor de valores faltantes')
plt.show()


# ### Eliminar filas con valores faltantes


# Eliminar filas que tienen todos los valores faltantes
df_cleaned = df.dropna(how='all')

# Ver las filas que tienen valores faltantes
print("Filas con valores faltantes:")
print(df[df.isnull().any(axis=1)])


# Registrar información sobre los datos eliminados
total_rows = len(df)
rows_after_cleaning = len(df_cleaned)
removed_rows = total_rows - rows_after_cleaning

print(f"Filas originales: {total_rows}")
print(f"Filas después de la limpieza: {rows_after_cleaning}")
print(f"Filas eliminadas: {removed_rows}")
print(f"Porcentaje de datos conservados: {(rows_after_cleaning/total_rows*100):.2f}%")


# ### Eliminar ID, IDvideo (machaco la variable df_cleaned)

# Eliminar la columna 'id' del DataFrame
df_cleaned = df_cleaned.drop(columns=['CommentId', 'VideoId'], axis=1)


# ### Convertir las etiquetas a valores binarios (0 para falso, 1 para verdadero)

from sklearn.preprocessing import LabelEncoder

# Crear un objeto LabelEncoder
le = LabelEncoder()

# Aplicar el LabelEncoder a las columnas especificadas
for col in ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 
             'IsObscene', 'IsHatespeech', 'IsRacist', 
             'IsNationalist', 'IsSexist', 'IsHomophobic', 
             'IsReligiousHate', 'IsRadicalism']:
    df_cleaned[col] = le.fit_transform(df_cleaned[col])



print(df_cleaned)


# ### Dividir en X e y

# Dividir en X e y
X = df_cleaned.drop('IsHatespeech', axis=1)  # Reemplaza 'tu_columna_objetivo' con el nombre de tu variable dependiente
y = df_cleaned['IsHatespeech']





# Dividir en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Voy a separar unos 10 registros para hacer una prueba luego.(e.g., 10 samples)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=10, random_state=42)


# ### Creación del Modelo

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# #### Vectorizar el texto usando TF-IDF


# Vectorizar el texto usando TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train['Text'])
X_test_vectorized = vectorizer.transform(X_test['Text'])



""" from sklearn.feature_extraction.text import CountVectorizer

# Suponiendo que 'text_column' es la columna que deseas vectorizar
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train['Text'])  # Asegúrate de que 'text_column' sea la columna correcta """



print("Forma de X_train_vectorized:", X_train_vectorized.shape)
print("Forma de y_train:", y_train.shape)


# Se utiliza TfidfVectorizer para convertir el texto en una representación numérica que puede ser utilizada por el modelo de aprendizaje automático.
# 
# · fit_transform(X_train) ajusta el vectorizador a los datos de entrenamiento y transforma el texto en una matriz de características TF-IDF (Term Frequency-Inverse Document Frequency).
# 
# · transform(X_test) aplica la misma transformación a los datos de prueba, utilizando el mismo ajuste que se realizó en el conjunto de entrenamiento.


# ####  Entrenar un modelo de regresión logística

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# Realizar predicciones
y_train_pred = model.predict(X_train_vectorized)
y_test_pred = model.predict(X_test_vectorized)


from sklearn.metrics import classification_report, accuracy_score
# Evaluar el rendimiento
print("Rendimiento en el conjunto de entrenamiento:")
print(classification_report(y_train, y_train_pred))
print("Precisión en el conjunto de entrenamiento:", accuracy_score(y_train, y_train_pred))



print("\nRendimiento en el conjunto de prueba:")
print(classification_report(y_test, y_test_pred))
print("Precisión en el conjunto de prueba:", accuracy_score(y_test, y_test_pred))



print(f'Sobreajuste:\n{round(accuracy_score(y_train, y_train_pred)-accuracy_score(y_test, y_test_pred),2)}')


# #### Guardar el entrenamiento del modelo y el vectorizador


# Guardar el modelo en un archivo
import pickle
with open('modelo_regLog.pkl', 'wb') as archivo:
    pickle.dump(model, archivo)


# Guardar el vectorizador en un archivo
with open('vectorizer.pkl', 'wb') as archivo:
    pickle.dump(vectorizer, archivo)


# #### Cargar el entrenamiento del modelo y el vectorizador

# Cargar el modelo y el vectorizador entrenados
with open('modelo_regLog.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


print(X_val)


# Función para clasificar un mensaje
def classify_message(text):
    # Transformar el texto en un vector de características
    X = vectorizer.transform([text])
    # Realizar la predicción utilizando el modelo entrenado
    y_pred = model.predict(X)[0]
    # Devolver el resultado de la clasificación
    return bool(y_pred)

# Iterar a través de los mensajes de validación
for i, message in enumerate(X_val['Text']):
    # Clasificar el mensaje y mostrar el resultado
    if classify_message(message):
        print(f"Los mensajes calificados como Hate son:\n'{message}'.\n\n")
    else:
        print(f"\nLos mensajes 'normales' son:\n'{message}'.")



# In[ ]:


#pip install streamlit-jupyter
#%streamlit run app.py no sirve

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
    return "Mensaje de odio" if y_pred == 1 else "Mensaje no ofensivo"


# Crear la aplicación Streamlit
st.title("Verificador de Mensajes Odiosos")
st.write("Ingresa un mensaje y te diré si es considerado odioso o no.")

# Obtener el mensaje del usuario
message = st.text_area("Mensaje a verificar:", height=100,key="message_input")

# Clasificar el mensaje y mostrar el resultado
if st.button("Verificar mensaje"):
    resultado = classify_message(message)
    st.write(resultado)