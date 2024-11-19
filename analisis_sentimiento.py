import pandas as pd
from transformers import pipeline, AutoTokenizer

# Inicializar el modelo de análisis de sentimientos y el tokenizador
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
sentiment_analyzer = pipeline('sentiment-analysis', model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Función para analizar el sentimiento de un texto con truncamiento basado en tokens
def analyze_sentiment(text):
    # Tokenizar el texto y truncar a 512 tokens
    tokens = tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
    truncated_text = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
    result = sentiment_analyzer(truncated_text)
    return result[0]['label'], result[0]['score']

# Función para procesar un DataFrame y agregar análisis de sentimientos
def process_dataframe(df, column_name):
    sentiment_labels = []
    sentiment_scores = []
    for tokens in df[column_name]:
        # Concatenar los tokens en una sola string
        text = ' '.join(tokens)
        label, score = analyze_sentiment(text)
        sentiment_labels.append(label)
        sentiment_scores.append(score)
    df['sentiment_label'] = sentiment_labels
    df['sentiment_score'] = sentiment_scores
    return df


# Cargar los datasets
train_df = pd.read_csv('/filtered_train_data.csv')
val_df = pd.read_csv('/filtered_val_data.csv')
test_df = pd.read_csv('/filtered_test_data.csv')

# Aplicar el análisis de sentimientos a cada dataset
train_df = process_dataframe(train_df, 'semantic_tokens')
val_df = process_dataframe(val_df, 'semantic_tokens')
test_df = process_dataframe(test_df, 'semantic_tokens')

# Guardar los DataFrames con los resultados del análisis de sentimientos
train_df.to_csv('/filtered_train_data_with_sentiment.csv', index=False)
val_df.to_csv('/filtered_val_data_with_sentiment.csv', index=False)
test_df.to_csv('/filtered_test_data_with_sentiment.csv', index=False)

print("Análisis de sentimientos completado y archivos guardados.")
