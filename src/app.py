import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


st.title("Fake or Real Summary Classifier")
st.write("This is a web app to classify text as fake (LLM generated summaries) or real.")

# Upload the model
model = joblib.load("../models/logistic_regression_tfidf_more_features.joblib")

with st.form(key='my_form'):
    text_input = st.text_area("Enter text to classify:")
    submit_button = st.form_submit_button("Classify")

if submit_button:
    # Preprocess the input text
    df = pd.DataFrame({'text': [text_input]})
    
    # engineer features
    df['text_length'] = df['text'].apply(len)
    df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(i) for i in x.split()]))

    prediction = model.predict(df)

    st.write("Prediction:", prediction)
    st.write("Confidence (0 for fake, and 1 for real):", model.predict_proba(df))