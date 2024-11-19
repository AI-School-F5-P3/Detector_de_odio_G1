import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('C:/4_F5/019_NPL/Detector_de_odio_G1/data/raw/youtoxic_english_1000.csv', sep=',')


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


rows_hate = df[df['IsHatespeech'] == True]
print(rows_hate.head())


print(df['IsHatespeech'].unique())


# Guardar df_cleaned
ruta='C:/4_F5/019_NPL/Detector_de_odio_G1/src/'
df_cleaned.to_csv(ruta+'df_cleaned.csv', index=False, sep=';',encoding='utf-8')

