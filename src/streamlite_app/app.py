import streamlit as st
import joblib
import pandas as pd
from transformers import BertTokenizer, RobertaTokenizer, RobertaForSequenceClassification, pipeline
import torch
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

# Configurar la API de YouTube
api_key = 'AIzaSyDbFLN95bAUlSFVayeIMhZQ8G6KxUeSCg4'
youtube = build('youtube', 'v3', developerKey=api_key)

# Cargar los modelos entrenados
model_bert = joblib.load('bert_model.pkl')
model_roberta = joblib.load('roberta_model.pkl')

# Cargar el modelo original de RoBERTa de Hugging Face
model_roberta_hf = RobertaForSequenceClassification.from_pretrained('roberta-base')

# Inicializar los tokenizers
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Inicializar el pipeline de análisis de sentimiento
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Función para tokenizar el texto
def tokenize_text(text, tokenizer):
    return tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )

# Función para predecir con el modelo seleccionado
def predict(text, model):
    if model == 'BERT':
        encodings = tokenize_text(text, bert_tokenizer)
        with torch.no_grad():
            outputs = model_bert(**encodings)
    elif model == 'RoBERTa':
        encodings = tokenize_text(text, roberta_tokenizer)
        with torch.no_grad():
            outputs = model_roberta(**encodings)
    elif model == 'RoBERTa HF':
        encodings = tokenize_text(text, roberta_tokenizer)
        with torch.no_grad():
            outputs = model_roberta_hf(**encodings)
    
    predictions = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(predictions, dim=1).item()
    return 'Hate Speech' if label == 1 else 'Not Hate Speech'

# Función para analizar el sentimiento
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']

# Función para obtener comentarios de YouTube
def get_youtube_comments(video_id, max_results=100):
    comments = []
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=max_results,
        textFormat='plainText'
    )
    response = request.execute()
    
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)
    
    return comments

# Función para obtener la transcripción del video de YouTube
def get_youtube_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = ' '.join([entry['text'] for entry in transcript])
    return transcript_text

# Interfaz de Streamlit
st.title('Detección de Mensajes de Odio en YouTube')
st.write('Introduce la URL de un video de YouTube para analizar los comentarios o el contenido del video.')

url_input = st.text_input('URL del video de YouTube')

# Selección del modelo
model_option = st.selectbox(
    'Selecciona el modelo para la clasificación:',
    ('BERT', 'RoBERTa', 'RoBERTa HF')
)

# Selección de la funcionalidad
function_option = st.selectbox(
    'Selecciona la funcionalidad:',
    ('Analizar Comentarios', 'Analizar Contenido del Video')
)

if st.button('Analizar'):
    if url_input:
        video_id = url_input.split('v=')[-1]
        
        # Incrustar el video de YouTube
        st.video(url_input)
        
        if function_option == 'Analizar Comentarios':
            max_comments = st.number_input('Número de comentarios a analizar', min_value=1, max_value=100, value=10)
            comments = get_youtube_comments(video_id, max_results=max_comments)
            
            for comment in comments:
                prediction = predict(comment, model_option)
                sentiment_label, sentiment_score = analyze_sentiment(comment)
                
                st.write(f'Comentario: {comment}')
                st.write(f'Predicción: {prediction}')
                st.write(f'Sentimiento: {sentiment_label} (Score: {sentiment_score:.2f})')
                
                if sentiment_label == 'POSITIVE':
                    st.image('https://path_to_happy_face_image.png', width=100)
                else:
                    st.image('https://path_to_sad_face_image.png', width=100)
        
        elif function_option == 'Analizar Contenido del Video':
            transcript_text = get_youtube_transcript(video_id)
            prediction = predict(transcript_text, model_option)
            sentiment_label, sentiment_score = analyze_sentiment(transcript_text)
            
            st.write(f'Transcripción del Video: {transcript_text}')
            st.write(f'Predicción: {prediction}')
            st.write(f'Sentimiento: {sentiment_label} (Score: {sentiment_score:.2f})')
            
            if sentiment_label == 'POSITIVE':
                st.image('https://path_to_happy_face_image.png', width=100)
            else:
                st.image('https://path_to_sad_face_image.png', width=100)
    else:
        st.write('Por favor, introduce una URL de YouTube.')
