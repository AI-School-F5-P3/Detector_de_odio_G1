
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from jinja2 import Environment, FileSystemLoader
import os
from sklearn.metrics import classification_report, accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier

df_code=pd.read_csv('C:/4_F5/019_NPL/Detector_de_odio_G1/src/df_code.csv', sep=';')


# Dividir en X e y
X = df_code.drop('IsHatespeech', axis=1)  # mi target
y = df_code['IsHatespeech']


# Dividir en conjuntos de entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Vectorizar el texto usando TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train['Text'])
X_test_vectorized = vectorizer.transform(X_test['Text'])


print("Forma de X_train_vectorized:", X_train_vectorized.shape)
print("Forma de y_train:", y_train.shape)


# Preparar datos

# load library
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
rfc = BalancedRandomForestClassifier()

# fit the predictor and target
model_rfc=rfc.fit(X_train_vectorized, y_train)



# Realizar predicciones
y_train_pred = model_rfc.predict(X_train_vectorized)
y_test_pred = model_rfc.predict(X_test_vectorized)



# Evaluar el rendimiento
print("Rendimiento en el conjunto de entrenamiento:")
print(classification_report(y_train, y_train_pred))
print("Precisión en el conjunto de entrenamiento:", accuracy_score(y_train, y_train_pred))



print("\nRendimiento en el conjunto de prueba:")
print(classification_report(y_test, y_test_pred))
print("Precisión en el conjunto de prueba:", accuracy_score(y_test, y_test_pred))




cm = confusion_matrix(y_test, y_test_pred)
print("Matriz de Confusión:")
print(cm)

# Guardar el modelo en un archivo
import pickle
with open('C:/4_F5/019_NPL/Detector_de_odio_G1/models/modelo_RFC.pkl', 'wb') as archivo:
    pickle.dump(model_rfc, archivo)

# Calcular métricas
train_report = classification_report(y_train, y_train_pred)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_report = classification_report(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
accuracy_diff = round((train_accuracy - test_accuracy)*100,2)

# Configuración de Jinja2
template_dir = "C:/4_F5/019_NPL/Detector_de_odio_G1/src"
env = Environment(loader=FileSystemLoader(template_dir))


#Usar el nombre relativo de la plantilla
template = env.get_template('template.html')
# Renderizar la plantilla con las métricas
output = template.render(
    train_accuracy=train_accuracy,
    train_report=train_report,
    test_accuracy=test_accuracy,
    test_report=test_report,
    accuracy_diff=accuracy_diff
)

# Guardar el informe como archivo HTML
with open('C:/4_F5/019_NPL/Detector_de_odio_G1/src/informe_modeloRFC.html', 'w', encoding='utf-8') as f:
    f.write(output)

print("Informe generado: informe_modeloRFC.html")