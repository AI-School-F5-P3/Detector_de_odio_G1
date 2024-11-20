import os
from dotenv import load_dotenv
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch

# Cargar las variables de entorno
load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')

# Configurar el dispositivo para usar la GPU si está disponible
device = 0 if torch.cuda.is_available() else -1

# Función para cargar y procesar un archivo CSV
def load_and_process_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Cargar los datasets
train_df = load_and_process_csv('/filtered_train_data_with_sentiment.csv')
val_df = load_and_process_csv('/filtered_val_data_with_sentiment.csv')
test_df = load_and_process_csv('/filtered_test_data_with_sentiment.csv')

# Calcular las probabilidades de cada valor en las columnas adicionales
def calculate_probabilities(df):
    probabilities = {}
    columns = ['IsHatespeech', 'IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 'IsObscene', 'IsRacist', 'IsNationalist', 'IsSexist', 'IsHomophobic', 'IsReligiousHate', 'IsRadicalism']
    for column in columns:
        probabilities[column] = df[column].mean()
    return probabilities

probabilities = calculate_probabilities(train_df)

# Ajustar las probabilidades para generar más datos con IsHatespeech como True
probabilities['IsHatespeech'] = 1.0  # Asegurar que todos los datos generados sean de odio

# Inicializar el modelo de generación de texto GPT-3
model_name = "EleutherAI/gpt-neo-2.7B"  # Usamos GPT-Neo como alternativa a GPT-3
text_generator = pipeline('text-generation', model=model_name, tokenizer=model_name, device=device)

# Función para generar texto sintético con parámetros ajustados
def generate_synthetic_text(seed_text, num_return_sequences=1):
    generated_texts = text_generator(seed_text, max_length=50, num_return_sequences=num_return_sequences, truncation=True, temperature=0.7, top_p=0.9, top_k=50)
    return [gen['generated_text'] for gen in generated_texts]

# Palabras semilla para generar diferentes tipos de odio
hate_words = {
    'IsToxic': ["idiot", "stupid", "moron"],
    'IsAbusive': ["abuse", "hit", "hurt"],
    'IsThreat': ["kill", "destroy", "attack"],
    'IsProvocative': ["provoke", "taunt", "mock"],
    'IsObscene': ["fuck", "shit", "bitch"],
    'IsRacist': ["nigger", "chink", "spic"],
    'IsNationalist': ["patriot", "nationalist", "traitor"],
    'IsSexist': ["slut", "whore", "bimbo"],
    'IsHomophobic': ["faggot", "dyke", "queer"],
    'IsReligiousHate': ["infidel", "heretic", "blasphemer"],
    'IsRadicalism': ["extremist", "radical", "terrorist"]
}

# Generar datos sintéticos para cada tipo de odio
synthetic_texts = {key: [] for key in hate_words.keys()}
for category, words in hate_words.items():
    for word in words:
        gen_texts = generate_synthetic_text(word, num_return_sequences=20)  # Ajustar la cantidad de textos generados
        synthetic_texts[category].extend(gen_texts)

# Crear DataFrames con los textos sintéticos
synthetic_dfs = {}
for category, texts in synthetic_texts.items():
    df = pd.DataFrame({'semantic_tokens': texts})
    df[category] = True
    synthetic_dfs[category] = df

# Combinar todos los DataFrames sintéticos
combined_synthetic_df = pd.concat(synthetic_dfs.values(), ignore_index=True)

# Asignar valores a las columnas adicionales basados en probabilidades
def assign_additional_columns(df, probabilities):
    for column, prob in probabilities.items():
        if column not in df.columns:
            df[column] = np.random.rand(len(df)) < prob
    return df

combined_synthetic_df = assign_additional_columns(combined_synthetic_df, probabilities)

# Inicializar el modelo de análisis de sentimientos
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device=device)

# Función para analizar el sentimiento de un texto
def analyze_sentiment(text):
    result = sentiment_analyzer(text, truncation=True)
    return result[0]['label'], result[0]['score']

# Aplicar el análisis de sentimientos a los datos sintéticos
combined_synthetic_df['sentiment_label'], combined_synthetic_df['sentiment_score'] = zip(*combined_synthetic_df['semantic_tokens'].apply(analyze_sentiment))

# Combinar los datasets originales con los datos sintéticos
combined_train_df = pd.concat([train_df, combined_synthetic_df], ignore_index=True)
combined_val_df = pd.concat([val_df, combined_synthetic_df], ignore_index=True)
combined_test_df = pd.concat([test_df, combined_synthetic_df], ignore_index=True)

# Guardar los DataFrames combinados
combined_train_df.to_csv('combined2_train_data_with_sentiment.csv', index=False)
combined_val_df.to_csv('combined2_val_data_with_sentiment.csv', index=False)
combined_test_df.to_csv('combined2_test_data_with_sentiment.csv', index=False)

print("Generación de datos sintéticos completada y archivos guardados.")
