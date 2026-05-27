import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import nltk

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# -------------------------------------------------------
# Download NLTK Stopwords
# -------------------------------------------------------
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# -------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------
st.set_page_config(
    page_title="Mental Health Sentiment Monitor",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
MAX_LENGTH = 50
VOCAB_SIZE = 78880
EMBEDDING_DIM = 100
RNN_UNITS = 128
NUM_CLASSES = 7

STOP_WORDS = set(stopwords.words('english'))

# -------------------------------------------------------
# Load Model + Tokenizer + Label Encoder
# -------------------------------------------------------
@st.cache_resource
def load_artifacts():

    try:

        # ---------------------------------------------------
        # Recreate Model Architecture
        # ---------------------------------------------------
        model = Sequential()

        # Embedding Layer
        model.add(
            Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=EMBEDDING_DIM
            )
        )

        # RNN Layer
        model.add(
            SimpleRNN(
                RNN_UNITS,
                activation='relu'
            )
        )

        # Output Layer
        model.add(
            Dense(
                NUM_CLASSES,
                activation='softmax'
            )
        )

        # ---------------------------------------------------
        # Build Model
        # ---------------------------------------------------
        dummy_input = np.zeros((1, MAX_LENGTH))

        model(dummy_input)

        # ---------------------------------------------------
        # Load Weights from .h5 file
        # ---------------------------------------------------
        model.load_weights(
            "sentiment_model_artifacts/rnn_sentiment_model.h5"
        )

        # ---------------------------------------------------
        # Load Tokenizer
        # ---------------------------------------------------
        with open(
            "sentiment_model_artifacts/tokenizer.pkl",
            "rb"
        ) as file:

            tokenizer = pickle.load(file)

        # ---------------------------------------------------
        # Load Label Encoder
        # ---------------------------------------------------
        with open(
            "sentiment_model_artifacts/label_encoder.pkl",
            "rb"
        ) as file:

            label_encoder = pickle.load(file)

        return model, tokenizer, label_encoder

    except Exception as e:

        st.error(f"Error loading artifacts:\n\n{e}")

        st.stop()

# -------------------------------------------------------
# Load Everything
# -------------------------------------------------------
model, tokenizer, label_encoder = load_artifacts()

# -------------------------------------------------------
# Text Preprocessing
# -------------------------------------------------------
def preprocess_text(text):

    text = str(text).lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    return text

# -------------------------------------------------------
# Prediction Function
# -------------------------------------------------------
def predict_sentiment(text):

    # Clean text
    cleaned_text = preprocess_text(text)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    # Padding
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding='post',
        truncating='post'
    )

    # Prediction
    prediction = model.predict(
        padded_sequence,
        verbose=0
    )

    # Highest probability index
    predicted_index = np.argmax(prediction)

    # Confidence score
    confidence = float(
        prediction[0][predicted_index]
    )

    # Convert label
    sentiment = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    return sentiment, confidence

# -------------------------------------------------------
# App Title
# -------------------------------------------------------
st.title("🧠 Mental Health Sentiment Monitoring System")

st.markdown("""
This AI-powered NLP application analyzes emotional sentiment using a trained Recurrent Neural Network (RNN).

### Supported Sentiments
- Normal
- Anxiety
- Depression
- Stress
- Suicidal
- Bipolar
- Personality disorder
""")

st.divider()

# -------------------------------------------------------
# User Input
# -------------------------------------------------------
user_input = st.text_area(
    "Enter your text below:",
    height=180,
    placeholder="Type your thoughts or feelings here..."
)

# -------------------------------------------------------
# Analyze Button
# -------------------------------------------------------
if st.button("Analyze Sentiment"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        with st.spinner("Analyzing sentiment..."):

            sentiment, confidence = predict_sentiment(
                user_input
            )

        st.success("Analysis Complete")

        # ---------------------------------------------------
        # Prediction Output
        # ---------------------------------------------------
        st.subheader("Predicted Sentiment")

        st.metric(
            label="Prediction",
            value=sentiment
        )

        # ---------------------------------------------------
        # Confidence Score
        # ---------------------------------------------------
        st.subheader("Confidence Score")

        st.progress(confidence)

        st.write(
            f"Confidence: {confidence:.2%}"
        )

        # ---------------------------------------------------
        # Emotional Alerts
        # ---------------------------------------------------
        distress_labels = [
            "depression",
            "stress",
            "anxiety",
            "suicidal",
            "bipolar",
            "personality disorder"
        ]

        if sentiment.lower() in distress_labels:

            st.error(
                f"Emotional distress ({sentiment}) detected. "
                "Please consider seeking support if needed."
            )

        elif sentiment.lower() == "normal":

            st.success(
                "Normal/Positive sentiment detected."
            )

        else:

            st.info(
                f"{sentiment} sentiment detected."
            )

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.divider()

st.caption(
    "Built with Streamlit, TensorFlow, NLTK, and Scikit-learn"
)