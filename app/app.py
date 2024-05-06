import streamlit as st
import joblib
from transformers import pipeline
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def local_css(file_path):
    with open("app/static/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# Initialize Transformers pipeline for text classification
model = pipeline('text-classification', model='bert-base-uncased', tokenizer='bert-base-uncased')

# Load models
model_files = ['app/models/SVM_model_Q1.pkl', 'app/models/SVM_model_Q2.pkl', 'app/models/SVM_model_Q3.pkl',
               'app/models/SVM_model_Q4.pkl', 'app/models/SVM_model_Q5.pkl']
models = [joblib.load(model_file) for model_file in model_files]

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=512)

# CEFR levels mapping
cefr_levels = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}

# Main application
st.image('app/static/image/cefr_logo.png')

# Login page
if 'login' not in st.session_state or not st.session_state['login']:
    st.title('Login')
    login_id = st.text_input('User ID:')
    if st.button('Login'):
        st.session_state['login'] = True
        st.session_state['user_id'] = login_id

# Quiz page
elif 'login' in st.session_state and st.session_state['login'] and 'predictions' not in st.session_state:
    st.title('Quiz')
    
    for i in range(1, 6):
        st.image(f'app/static/image/{i}.jpg')
        answer = st.text_area(f'Question {i}: Describe the picture shown.', '')

    if st.button('Predict'):
        answers = [answer1, answer2, answer3, answer4, answer5]
        if all(answers):
            # Tokenize and predict CEFR levels for each answer
            embeddings = [model(answer)[0]['label'] for answer in answers]
            predictions = [model.predict([emb])[0] for model, emb in zip(models, embeddings)]
            total_score = sum([cefr_levels[pred] for pred in predictions])
            final_cefr_level = determine_cefr_level(total_score)

            st.session_state['predictions'] = predictions
            st.session_state['total_score'] = total_score
            st.session_state['final_cefr_level'] = final_cefr_level

            st.experimental_rerun()

# Completion page
elif 'predictions' in st.session_state:
    st.title(f'User ID: {st.session_state["user_id"]}')
    st.write('---')
    for i, prediction in enumerate(st.session_state['predictions'], start=1):
        st.write(f'Predicted CEFR level for description {i}: {prediction}')
    st.write('---')
    st.title(f'Total CEFR Level: {st.session_state["final_cefr_level"]}')
    st.write('---')
    if st.button('Reset'):
        st.session_state.pop('predictions', None)
        st.session_state.pop('total_score', None)
        st.session_state.pop('final_cefr_level', None)
        st.experimental_rerun()
