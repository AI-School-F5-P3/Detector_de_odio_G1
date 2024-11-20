import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from pandas_profiling import ProfileReport
from ydata_profiling import ProfileReport #para Python 3.12

# Cargar el dataset
df = pd.read_csv('C:/4_F5/019_NPL/Detector_de_odio_G1/data/raw/youtoxic_english_1000.csv', sep=',')


# Generar un reporte
profile_hate= ProfileReport(df,title="Reporte Exploratorio")
ruta="C:/4_F5/019_NPL/Detector_de_odio_G1/exploracion/"
profile_hate.to_file(ruta+"reporte_hate.html") 



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
