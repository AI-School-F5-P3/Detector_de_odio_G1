import streamlit as st
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import os
import logging
import sqlite3
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv('YOUTUBE_API_KEY')

# Validate API key
if not api_key:
    st.error("YouTube API key not found. Please set the YOUTUBE_API_KEY environment variable.")
    st.stop()

# Configure the YouTube API
try:
    youtube = build('youtube', 'v3', developerKey=api_key)
except Exception as e:
    st.error(f"Error initializing YouTube API: {e}")
    st.stop()


def load_model(model_path=None):
    try:
        if not model_path:
            model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        else:
            model = RobertaForSequenceClassification.from_pretrained(
                'roberta-base',
                state_dict=torch.load(model_path, map_location='cpu')
            )
        return model
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        st.error(f"Could not load model: {e}")
        return None


def tokenize_text(text, tokenizer, max_length=512):
    try:
        return tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    except Exception as e:
        logger.error(f"Tokenization error: {e}")
        return None


def split_text_into_batches(text, max_length=512):
    tokens = text.split()
    batches = [' '.join(tokens[i:i + max_length]) for i in range(0, len(tokens), max_length)]
    return batches


def predict_hate_speech_batches(text, model, tokenizer):
    batches = split_text_into_batches(text)
    predictions = []
    for batch in batches:
        prediction = predict_hate_speech(batch, model, tokenizer)
        predictions.append(prediction)
    return predictions


def predict_hate_speech(text, model, tokenizer):
    try:
        encodings = tokenize_text(text, tokenizer)
        if encodings is None:
            return "Error in tokenization"

        with torch.no_grad():
            outputs = model(**encodings)

        predictions = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(predictions, dim=1).item()
        return 'Hate Speech' if label == 1 else 'Not Hate Speech'
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Prediction Error"


def get_youtube_comments(video_id, max_results=10):
    try:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=max_results,
            textFormat='plainText'
        )
        response = request.execute()

        comments = [
            item['snippet']['topLevelComment']['snippet']['textDisplay']
            for item in response.get('items', [])
        ]
        return comments
    except Exception as e:
        logger.error(f"YouTube comments retrieval error: {e}")
        st.error("Could not retrieve comments. Check video availability.")
        return []


def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript])
    except Exception as e:
        logger.error(f"Transcript retrieval error: {e}")
        st.error("Could not retrieve video transcript.")
        return ""


def save_to_database(data):
    """
    Save prediction results to SQLite database
    """
    try:
        # Conexión a la base de datos
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()

        # Crear la tabla si no existe
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id INTEGER PRIMARY KEY, comment TEXT, prediction TEXT, sentiment TEXT, score REAL)''')

        # Convertir prediction (lista) a cadena JSON
        prediction_json = json.dumps(data['prediction'])

        # Insertar los datos en la base de datos
        c.execute("INSERT INTO predictions (comment, prediction, sentiment, score) VALUES (?, ?, ?, ?)",
                  (data['comment'], prediction_json, data['sentiment'], data['score']))

        # Confirmar los cambios
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
    finally:
        conn.close()


def main():
    page_bg_img = '''
    <style>
    body {
        background-image: url("C:/Users/samir/Detector de odio/Detector_de_odio_G1/src/streamlite_app/matrix-2953869_1280.jpg");
        background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.image("C:/Users/samir/Detector de odio/Detector_de_odio_G1/src/streamlite_app/image_toxic_tube-removebg-preview.png", use_container_width=True)
    st.title('TOXICTUBE')

    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model_roberta_pretrained = load_model('C:/Users/samir/Detector de odio/Detector_de_odio_G1/src/models/roberta_model.pth')
    model_roberta_hf = load_model()
    sentiment_analyzer = pipeline('sentiment-analysis')

    st.markdown("### Enter YouTube Video URL")
    url_input = st.text_input('YouTube Video URL')
    st.markdown("### Select Classification Model")
    model_option = st.selectbox(
        'Select Classification Model',
        ['RoBERTa Pretrained', 'RoBERTa HF']
    )
    st.markdown("### Select Analysis Type")
    function_option = st.selectbox(
        'Analysis Type',
        ['Analyze Comments', 'Analyze Video Content']
    )

    if st.button('Analyze'):
        if not url_input:
            st.warning('Please enter a YouTube video URL')
            return

        video_id = url_input.split('v=')[-1]

        st.video(url_input)

        active_model = model_roberta_pretrained if model_option == 'RoBERTa Pretrained' else model_roberta_hf

        if active_model is None:
            st.error("No model loaded. Please check the model path and try again.")
            return

        if function_option == 'Analyze Comments':
            comments = get_youtube_comments(video_id)
            for comment in comments:
                hate_predictions = predict_hate_speech_batches(comment, active_model, roberta_tokenizer)
                sentiment = sentiment_analyzer(comment)[0]

                st.write(f"**Comment:** {comment}")
                st.write(f"**Hate Speech Predictions:** {hate_predictions}")
                st.write(f"**Sentiment:** {sentiment['label']} (Score: {sentiment['score']:.2f})")
                save_to_database({
                    'comment': comment,
                    'prediction': hate_predictions,
                    'sentiment': sentiment['label'],
                    'score': sentiment['score']
                })

        elif function_option == 'Analyze Video Content':
            transcript = get_youtube_transcript(video_id)
            if transcript:
                hate_predictions = predict_hate_speech_batches(transcript, active_model, roberta_tokenizer)
                sentiment = sentiment_analyzer(transcript)[0]

                st.write(f"**Video Transcript:** {transcript}")
                st.write(f"**Hate Speech Predictions:** {hate_predictions}")
                st.write(f"**Sentiment:** {sentiment['label']} (Score: {sentiment['score']:.2f})")
                save_to_database({
                    'comment': transcript,
                    'prediction': hate_predictions,
                    'sentiment': sentiment['label'],
                    'score': sentiment['score']
                })


if __name__ == "__main__":
    main()
