# 🔍 Detección de Mensajes de Odio con Machine Learning

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)

## 📖 Descripción

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning capaz de predecir si un mensaje está clasificado como **"hate" (mensaje de odio)**. Para ello:

- Se utiliza un **dataset** de mensajes etiquetados como "hate" o "not hate".
- Se probaron y evaluaron diversos modelos de Machine Learning.
- Se implementó una interfaz interactiva con **Streamlit** para probar la funcionalidad en tiempo real.

La aplicación permite cargar mensajes, analizar la predicción y visualizar estadísticas de rendimiento.

---

## 🚀 Características

- **Interfaz fácil de usar:** Desarrollada con Streamlit para la clasificación de mensajes.
- **Soporte para múltiples modelos:** Compara el rendimiento de diferentes algoritmos de Machine Learning.
- **Análisis interactivo:** Muestra estadísticas y visualizaciones de las predicciones.
- **Preprocesamiento de datos automático:** Limpieza y preparación del texto para los modelos.

---

## 🛠️ Instalación y Uso

### 1️⃣ Clonar el repositorio: 

```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
```

### 2️⃣ Instalar dependencias:
Es recomendable usar un entorno virtual:
python -m venv venv
source venv/bin/activate
En Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3️⃣ Ejecutar la aplicación:

Ejecuta el comando:
```bash
streamlit run app.py
```
Abre tu navegador en http://localhost:8501.

---
## 📂 Estructura del Proyecto


### Directory Descriptions

- `data/`: Contains the dataset files used for training and testing the models.
  
- `models/`: Contains the trained models that can be used for predictions or further analysis.

- `app.py`: The main code for the Streamlit application that provides the user interface for interacting with the models.

- `requirements.txt`: Lists the dependencies required to run the project. Use this file to install the necessary packages.

- `README.md`: This file itself, providing an overview of the project structure.

- `notebooks/`: Contains Jupyter notebooks for experiments, data analysis, and visualization.

---
## Installation

Para levantar el entorno e instalar las dependencias necesarias:
Nota: Para cada tipo de aplicación se necesitaron distintos requiements.

```bash
pip install -r requirements.txt
```

---
## 📊 Evaluación de Modelos
Se probaron los siguientes algoritmos de Machine Learning:

- Regresión Logística
- Random Forest
- Redes Neuronales (Transformers).


Los resultados se evaluaron utilizando métricas como:
- Precisión (Accuracy)
- Puntaje F1
- Overfitting

----
## 📦 Dataset
El dataset utilizado contiene mensajes clasificados como:

- `hate (odio)`: Mensajes con lenguaje ofensivo o discriminatorio.
- `not hate (no odio)`: Mensajes neutrales o positivos.

El preprocesamiento incluye:
- Eliminación de stopwords.
- Tokenización.
- Limpieza de caracteres especiales.

---
## 📌 Funcionalidades de la Aplicación
- Cargar mensajes: Proporciona un texto para análisis.
- Visualizar predicción: La aplicación clasifica el mensaje como "hate" o "not hate".
- Estadísticas interactivas: Muestra métricas y distribuciones basadas en los datos analizados.

---
## 🧠 Modelos y Tecnologías Utilizadas
- Machine Learning: Scikit-learn, Transformers (HuggingFace).
- Procesamiento de texto: NLTK, SpaCy.
- Interfaz: Streamlit.

---
## 🏗️ Mejoras Futuras
- Entrenamiento de modelos con datasets más grandes y diversos.
- Optimización del rendimiento para grandes volúmenes de datos.
- Implementación de explicabilidad del modelo (XAI).

---
---
## 📄 Licencia
Este proyecto está licenciado bajo la licencia MIT. Consulta el archivo LICENSE para más información.


