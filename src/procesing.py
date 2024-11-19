import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# Cargar el dataset
df_cleaned = pd.read_csv('C:/4_F5/019_NPL/Detector_de_odio_G1/src/df_cleaned.csv', sep=',')


# Eliminar la columna 'id' del DataFrame
df_cleaned = df_cleaned.drop(columns=['CommentId', 'VideoId'], axis=1)


from sklearn.preprocessing import LabelEncoder

# Crear un objeto LabelEncoder
le = LabelEncoder()

# Aplicar el LabelEncoder a las columnas especificadas
for col in ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 
             'IsObscene', 'IsHatespeech', 'IsRacist', 
             'IsNationalist', 'IsSexist', 'IsHomophobic', 
             'IsReligiousHate', 'IsRadicalism']:
    df_cleaned[col] = le.fit_transform(df_cleaned[col])

print(df_cleaned[df_cleaned['IsHatespeech']==True])
rows_hate=df_cleaned[df_cleaned['IsHatespeech']==True]

rows_hate.to_csv('C:/4_F5/019_NPL/Detector_de_odio_G1/src/rows_hate.csv', index=False, sep=';',encoding='utf-8')


