
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_code=pd.read_csv('C:/4_F5/019_NPL/Detector_de_odio_G1/src/df_code.csv', sep=';')


# Dividir en X e y
X = df_code.drop('IsHatespeech', axis=1)  # mi target
y = df_code['IsHatespeech']


# Dividir en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score



# Vectorizar el texto usando TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train['Text'])
X_test_vectorized = vectorizer.transform(X_test['Text'])


print("Forma de X_train_vectorized:", X_train_vectorized.shape)
print("Forma de y_train:", y_train.shape)
