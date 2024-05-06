import streamlit as st
import joblib
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def local_css(file_path):
    with open("app/static/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")


# Initialize tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load models
model1 = joblib.load('app/models/SVM_model_Q1.pkl')
model2 = joblib.load('app/models/SVM_model_Q2.pkl')
model3 = joblib.load('app/models/SVM_model_Q3.pkl')
model4 = joblib.load('app/models/SVM_model_Q4.pkl')
model5 = joblib.load('app/models/SVM_model_Q5.pkl')

# ตั้งค่า TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=768)

# คำนวณคะแนนรวมสำหรับแต่ละระดับ CEFR
cefr_levels = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}

# Main application
st.image('app/static/image/cefr_logo.png')

# Login page
if 'login' not in st.session_state or not st.session_state['login']:
    st.title('Login')
    login_id = st.text_input('User ID:')
    if st.button('Login'):
        # ตรวจสอบผู้ใช้
        st.success('Logged in successfully!')
        st.session_state['login'] = True
        st.session_state['user_id'] = login_id  # จัดเก็บ ID ผู้ใช้ในเซสชัน

# Quiz page
elif 'login' in st.session_state and st.session_state['login'] and 'predictions' not in st.session_state:
    st.title('Quiz')
    
    st.image('app/static/image/1.jpg')
    answer1 = st.text_area('Question 1: Describe the picture shown.', '')
    st.image('app/static/image/2.jpg')
    answer2 = st.text_area('Question 2: Describe the picture shown.', '')
    st.image('app/static/image/3.jpg')
    answer3 = st.text_area('Question 3: Describe the picture shown.', '')
    st.image('app/static/image/4.jpg')
    answer4 = st.text_area('Question 4: Describe the picture shown.', '')
    st.image('app/static/image/5.jpg')
    answer5 = st.text_area('Question 5: Describe the picture shown.', '')

    if st.button('Predict'):
        answers = [answer1, answer2, answer3, answer4, answer5]
        if all(answers):
            embeddings = []

            # Tokenize และ embeddings สำหรับแต่ละคำตอบ
            for answer in answers:
                encoded_input = tokenizer(answer, return_tensors='pt', padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    output = bert_model(**encoded_input)
                embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)

            # ทำนายระดับ CEFR สำหรับแต่ละคำตอบ
            predictions = [model.predict([emb])[0] for model, emb in zip([model1, model2, model3, model4, model5], embeddings)]

            # คำนวณคะแนนรวม
            total_scores = [cefr_levels[pred] for pred in predictions]
            total_score = sum(total_scores)

            # กำหนดระดับ CEFR
            def determine_cefr_level(score):
                if score >= 26:
                    return 'C2'
                elif score >= 21:
                    return 'C1'
                elif score >= 16:
                    return 'B2'
                elif score >= 11:
                    return 'B1'
                elif score >= 6:
                    return 'A2'
                else:
                    return 'A1'

            final_cefr_level = determine_cefr_level(total_score)

            # เก็บผลลัพธ์ไว้ในเซสชัน
            st.session_state['predictions'] = predictions
            st.session_state['total_score'] = total_score
            st.session_state['final_cefr_level'] = final_cefr_level

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
