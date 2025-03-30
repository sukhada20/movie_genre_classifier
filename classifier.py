import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load the trained model and vectorizer
with open("svm_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

st.title("Movie Genre Predictor")
st.write("Enter a movie description to predict its genre.")

# Text input for movie description
description = st.text_area("Movie Description", "")

if st.button("Predict Genre"):
    if description:
        # Transform input text
        input_vector = vectorizer.transform([description])
        # Predict genre
        prediction = model.predict(input_vector)
        st.success(f"Predicted Genre: {prediction[0]}")
    else:
        st.error("Please enter a movie description.")
