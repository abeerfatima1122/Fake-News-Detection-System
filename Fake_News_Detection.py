import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models and vectorizers
@st.cache_resource
def load_models():
    naive_bayes = pickle.load(open('naive_bayes_model.sav', 'rb'))
    random_forest = pickle.load(open('random_forest_model.sav', 'rb'))
    tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
    lstm_model = tf.keras.models.load_model('lstm_model.h5')
    tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pickle', 'rb'))
    return naive_bayes, random_forest, tokenizer, lstm_model, tfidf_vectorizer

naive_bayes, random_forest, tokenizer, lstm_model, tfidf_vectorizer = load_models()

# Function to preprocess input text
def preprocess_text(text, method='tfidf'):
    if method == 'tfidf':
        return tfidf_vectorizer.transform([text])
    elif method == 'tokenizer':
        sequence = tokenizer.texts_to_sequences([text])
        return pad_sequences(sequence, maxlen=100)  # Adjust maxlen as per training

# Streamlit UI
st.title("Fake News Detection")
st.write("Enter text to classify whether it's Fake News or Real News.")

user_input = st.text_area("Enter news text:")
model_choice = st.selectbox("Select Model:", ["Naive Bayes", "Random Forest", "LSTM"])

if st.button("Predict"):
    if user_input:
        if model_choice == "Naive Bayes":
            processed_text = preprocess_text(user_input, method='tfidf')
            prediction = naive_bayes.predict(processed_text)
        elif model_choice == "Random Forest":
            processed_text = preprocess_text(user_input, method='tfidf')
            prediction = random_forest.predict(processed_text)
        elif model_choice == "LSTM":
            processed_text = preprocess_text(user_input, method='tokenizer')
            prediction = lstm_model.predict(processed_text)
            prediction = np.argmax(prediction, axis=1)  # Convert probabilities to class labels
        
        # Mapping prediction to meaning
        label_map = {0: "Fake News", 1: "Real News"}
        st.success(f"Predicted Class: {label_map[prediction[0]]}")
    else:
        st.warning("Please enter some text for prediction.")

# Run the app using: streamlit run filename.py