import pandas as pd
import os

class TextPreprocessor:
    def __init__(self):
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        import spacy
        
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load('en_core_web_sm')

    def clean_text(self, text):
        """Limpia y normaliza el texto."""
        import re
        # Convertir a mayúsculas
        text = text.upper()
        # Eliminar URLs
        text = re.sub(r'HTTPS?://\S+|WWW\.\S+', '', text)
        # Eliminar caracteres especiales y números
        text = re.sub(r'[^A-Z\s]', '', text)
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_semantic_tokens(self, text):
        """Extrae tokens semánticamente significativos usando spaCy."""
        doc = self.nlp(text.lower())
        tokens = [token.lemma_.upper() for token in doc 
                  if (token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']) 
                  and (token.lemma_.lower() not in self.stop_words)]
        return tokens

    def process_dataframe(self, df, text_column):
        """Procesa un DataFrame para limpiar y generar tokens semánticos."""
        if text_column not in df.columns:
            raise ValueError(f"La columna '{text_column}' no existe en el DataFrame. Columnas disponibles: {df.columns}")
        
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        df['semantic_tokens'] = df['cleaned_text'].apply(self.get_semantic_tokens)
        df['processed_text'] = df['semantic_tokens'].apply(' '.join)
        
        return df

# Ruta del archivo
data_dir = os.path.join("data", "processed")
train_data_path = os.path.join(data_dir, "train_data1.csv")
val_data_path = os.path.join(data_dir, "val_data1.csv")
test_data_path = os.path.join(data_dir, "test_data1.csv")

# Cargar datasets
train_data1 = pd.read_csv(train_data_path)
val_data1 = pd.read_csv(val_data_path)
test_data1 = pd.read_csv(test_data_path)

# Crear instancia del preprocesador
preprocessor = TextPreprocessor()

# Procesar los datasets
train_data1 = preprocessor.process_dataframe(train_data1, text_column='semantic_tokens')
val_data1 = preprocessor.process_dataframe(val_data1, text_column='semantic_tokens')
test_data1 = preprocessor.process_dataframe(test_data1, text_column='semantic_tokens')

# Guardar los datasets procesados
train_data1.to_csv(os.path.join(data_dir, "train_data1_processed.csv"), index=False)
val_data1.to_csv(os.path.join(data_dir, "val_data1_processed.csv"), index=False)
test_data1.to_csv(os.path.join(data_dir, "test_data1_processed.csv"), index=False)

print("Preprocesamiento completado. Archivos guardados.")
