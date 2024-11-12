import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import spacy

class TextPreprocessor:
    def __init__(self):
        # Descargar recursos necesarios de NLTK
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Cargar modelo de spaCy para análisis semántico
        self.nlp = spacy.load('en_core_web_sm')
    
    def clean_text(self, text):
        """Limpia y normaliza el texto"""
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
        """Extrae tokens semánticamente significativos usando spaCy"""
        doc = self.nlp(text.lower())
        # Mantener solo sustantivos, verbos, adjetivos y adverbios
        tokens = [token.lemma_.upper() for token in doc 
                 if (token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']) 
                 and (token.lemma_.lower() not in self.stop_words)]
        return tokens
    
    def process_dataframe(self, df):
        """Procesa todo el DataFrame"""
        # Crear copia del DataFrame
        processed_df = df.copy()
        
        # Procesar textos
        processed_df['cleaned_text'] = processed_df['Text'].apply(self.clean_text)
        processed_df['semantic_tokens'] = processed_df['cleaned_text'].apply(self.get_semantic_tokens)
        
        # Unir tokens en un solo texto
        processed_df['processed_text'] = processed_df['semantic_tokens'].apply(' '.join)
        
        return processed_df
