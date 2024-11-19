#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# In[132]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[133]:


# Cargar el dataset
df = pd.read_csv('C:/4_F5/019_NPL/Detector_de_odio_G1/youtoxic_english_1000.csv', sep=',')


# In[134]:


#from pandas_profiling import ProfileReport
#from ydata_profiling import ProfileReport #para Python 3.12


# In[135]:


# Generar un reporte

""" profile_hate= ProfileReport(df,title="Reporte Exploratorio")
ruta="C:/4_F5/019_NPL/Detector_de_odio_G1/"
profile_hate.to_file(ruta+"reporte_hate.html") """


# In[136]:


# Obtener información general sobre el conjunto de datos
print(df.info())


# In[137]:


print(df.head())


# In[138]:


# Verificar la distribución de las etiquetas
print(df.sum())


# In[139]:


# Analizar valores faltantes
print("Cantidad de valores faltantes por columna:")
print(df.isnull().sum())


# In[140]:


# Calcular el porcentaje de valores faltantes por columna
print("\nPorcentaje de valores faltantes por columna:")
print((df.isnull().sum() / len(df) * 100).round(2))


# In[141]:


# Visualizar valores faltantes
#using a Jupyter Notebook to enable inline plotting needs this--> %matplotlib inline 
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Mapa de calor de valores faltantes')
plt.show()


# ### Eliminar filas con valores faltantes

# In[142]:


# Eliminar filas que tienen todos los valores faltantes
df_cleaned = df.dropna(how='all')


# In[143]:


# Ver las filas que tienen valores faltantes
print("Filas con valores faltantes:")
print(df[df.isnull().any(axis=1)])


# In[144]:


# Registrar información sobre los datos eliminados
total_rows = len(df)
rows_after_cleaning = len(df_cleaned)
removed_rows = total_rows - rows_after_cleaning

print(f"Filas originales: {total_rows}")
print(f"Filas después de la limpieza: {rows_after_cleaning}")
print(f"Filas eliminadas: {removed_rows}")
print(f"Porcentaje de datos conservados: {(rows_after_cleaning/total_rows*100):.2f}%")


# In[145]:


rows_hate = df[df['IsHatespeech'] == 'True']
print(rows_hate)



# Limpiar la columna 'Text'
df_cleaned['Text'] = df_cleaned['Text'].str.replace(r'\s*\n\s*', ' ', regex=True)  # Reemplaza saltos de línea por un espacio
df_cleaned['Text'] = df_cleaned['Text'].str.strip()  # Elimina espacios al inicio y al final


# In[146]:


print(df['IsHatespeech'].unique())


# ### Eliminar ID IDvideo (machaco la variable df_cleaned)

# In[147]:


# Eliminar la columna 'id' del DataFrame
df_cleaned = df_cleaned.drop(columns=['CommentId', 'VideoId'], axis=1)


# ### Convertir las etiquetas a valores binarios (0 para falso, 1 para verdadero)

# In[148]:


from sklearn.preprocessing import LabelEncoder

# Crear un objeto LabelEncoder
le = LabelEncoder()

# Aplicar el LabelEncoder a las columnas especificadas
for col in ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 
             'IsObscene', 'IsHatespeech', 'IsRacist', 
             'IsNationalist', 'IsSexist', 'IsHomophobic', 
             'IsReligiousHate', 'IsRadicalism']:
    df_cleaned[col] = le.fit_transform(df_cleaned[col])


# In[149]:


print(df_cleaned)


# In[150]:


print(df_cleaned[df_cleaned['IsHatespeech']==True])
rows_hate=df_cleaned[df_cleaned['IsHatespeech']==True]

rows_hate.to_csv('rows_hate.csv', index=False, sep=';',encoding='utf-8')


# ### Dividir en X e y

# In[151]:


# Dividir en X e y
X = df_cleaned.drop('IsHatespeech', axis=1)  # Reemplaza 'tu_columna_objetivo' con el nombre de tu variable dependiente
y = df_cleaned['IsHatespeech']


# In[ ]:






# In[152]:


# Dividir en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:





# In[ ]:





# In[153]:


# Voy a separar unos 10 registros para hacer una prueba luego.(e.g., 10 samples)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=10, random_state=42)


# ### Creación del Modelo

# In[154]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# #### Vectorizar el texto usando TF-IDF

# In[155]:


# Vectorizar el texto usando TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train['Text'])
X_test_vectorized = vectorizer.transform(X_test['Text'])


# In[156]:


""" from sklearn.feature_extraction.text import CountVectorizer

# Suponiendo que 'text_column' es la columna que deseas vectorizar
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train['Text'])  # Asegúrate de que 'text_column' sea la columna correcta """


# In[157]:


print("Forma de X_train_vectorized:", X_train_vectorized.shape)
print("Forma de y_train:", y_train.shape)


# Se utiliza TfidfVectorizer para convertir el texto en una representación numérica que puede ser utilizada por el modelo de aprendizaje automático.
# 
# · fit_transform(X_train) ajusta el vectorizador a los datos de entrenamiento y transforma el texto en una matriz de características TF-IDF (Term Frequency-Inverse Document Frequency).
# 
# · transform(X_test) aplica la misma transformación a los datos de prueba, utilizando el mismo ajuste que se realizó en el conjunto de entrenamiento.

# In[158]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
# Aplicar SMOTE para aumentar la clase minoritaria
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vectorized, y_train)

# Verificar la nueva distribución
from collections import Counter
print("Distribución original:", Counter(y_train))
print("Distribución después de SMOTE:", Counter(y_train_resampled))


# ####  Entrenar un modelo de regresión logística

# In[187]:


model = LogisticRegression(class_weight='balanced',penalty='l2', C=0.1, random_state=42)
model.fit(X_train_resampled, y_train_resampled)


# In[188]:


# Realizar predicciones
y_train_pred = model.predict(X_train_resampled)
y_test_pred = model.predict(X_test_vectorized)


# In[189]:


from sklearn.metrics import classification_report, accuracy_score
# Evaluar el rendimiento
print("Rendimiento en el conjunto de entrenamiento:")
print(classification_report(y_train_resampled, y_train_pred))
print("Precisión en el conjunto de entrenamiento:", accuracy_score(y_train_resampled, y_train_pred))


# In[190]:


print("\nRendimiento en el conjunto de prueba:")
print(classification_report(y_test, y_test_pred))
print("Precisión en el conjunto de prueba:", accuracy_score(y_test, y_test_pred))


# In[191]:


print(accuracy_score(y_train_resampled, y_train_pred)-accuracy_score(y_test, y_test_pred))


# In[164]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

cm = confusion_matrix(y_test, y_test_pred)
print("Matriz de Confusión:")
print(cm)


# #### Entrenar Modelo con XGBoost

# In[165]:


from xgboost import XGBClassifier

# Ajustar scale_pos_weight para equilibrar el modelo
model_xg = XGBClassifier(scale_pos_weight=(len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1])))
model_xg.fit(X_train_resampled, y_train_resampled)


# 

# In[166]:


# Realizar predicciones
y_train_pred_xg= model_xg.predict(X_train_resampled)
y_test_pred_xg= model_xg.predict(X_test_vectorized)


# In[167]:


from sklearn.metrics import classification_report, accuracy_score
# Evaluar el rendimiento
print("Rendimiento en el conjunto de entrenamiento en XG:")
print(classification_report(y_train_resampled, y_train_pred_xg))
print("Precisión en el conjunto de entrenamiento XG:", accuracy_score(y_train_resampled, y_train_pred_xg))


# In[176]:


print("\nRendimiento en el conjunto de prueba de XG:")
print(classification_report(y_test, y_test_pred_xg))
print("Precisión en el conjunto de prueba de XG:", accuracy_score(y_test, y_test_pred_xg))


# In[177]:


print(accuracy_score(y_train_resampled, y_train_pred_xg)-accuracy_score(y_test, y_test_pred_xg))


# In[172]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

cm = confusion_matrix(y_test, y_test_pred_xg)
print("Matriz de Confusión:")
print(cm)


# #### Modelo Random Classifier

# In[ ]:


# load library
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

# fit the predictor and target
model_rcf=rfc.fit(X_train_resampled, y_train_resampled)

# predict
y_train_pred_rfc= model_rcf.predict(X_train_resampled)
y_test_pred_rfc= model_rcf.predict(X_test_vectorized)



# In[179]:


from sklearn.metrics import classification_report, accuracy_score
# Evaluar el rendimiento
print("Rendimiento en el conjunto de entrenamiento en Random Classifier:")
print(classification_report(y_train_resampled, y_train_pred_rfc))
print("Precisión en el conjunto de entrenamiento Random Classifier:", accuracy_score(y_train_resampled, y_train_pred_rfc))


# In[180]:


print("\nRendimiento en el conjunto de prueba de Random Classifier:")
print(classification_report(y_test, y_test_pred_rfc))
print("Precisión en el conjunto de prueba de Random Classifer:", accuracy_score(y_test, y_test_pred_xg))


# In[175]:


print(accuracy_score(y_train_resampled, y_train_pred_rfc)-accuracy_score(y_test, y_test_pred_rfc))


# #### Guardar el entrenamiento del modelo y el vectorizador

# In[ ]:


# Guardar el modelo en un archivo
import pickle
with open('modelo_regLog.pkl', 'wb') as archivo:
    pickle.dump(model, archivo)


# In[ ]:


# Guardar el vectorizador en un archivo
with open('vectorizer.pkl', 'wb') as archivo:
    pickle.dump(vectorizer, archivo)


# #### Cargar el entrenamiento del modelo y el vectorizador

# In[ ]:


#Es para cargar el modelo

import pickle

# Cargar el modelo y el vectorizador entrenados
with open('modelo_regLog.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# In[ ]:


print(X_val)


# In[192]:


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
    if classify_message(message)==1:
        print(f"Los mensajes calificados como Hate son:\n'{message}'.\n\n")
    else:
        print(f"\nLos mensajes 'normales' son:\n'{message}'.")


# In[ ]:

# Función para clasificar un mensaje
def classify_message(text):
    # Transformar el texto en un vector de características
    X = vectorizer.transform([text])
    # Realizar la predicción utilizando el modelo entrenado
    y_pred = model.predict(X)[0]
    # Devolver el resultado de la clasificación
    return "Mensaje de odio" if y_pred == 1 else "Mensaje no ofensivo"

import streamlit as st
# Crear la aplicación Streamlit
st.title("Verificador de Mensajes Odiosos")
st.write("Ingresa un mensaje y te diré si es considerado odioso o no.")

# Obtener el mensaje del usuario
message = st.text_area("Mensaje a verificar:", height=100,key="message_input")

# Clasificar el mensaje y mostrar el resultado
if st.button("Verificar mensaje"):
    resultado = classify_message(message)
    st.write(resultado)


