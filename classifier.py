import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load pre-trained model and vectorizer
with open("svm_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Ensure NLTK resources are downloaded
nltk.download("stopwords")
nltk.download("punkt")

# Text cleaning function
def clean_text(text):
    stemmer = LancasterStemmer()
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = nltk.word_tokenize(text)
    text = " ".join([stemmer.stem(word) for word in words if word not in stop_words and len(word) > 2])
    return text

# Streamlit UI
st.title("🎬 Movie Genre Predictor")
st.write("Enter a movie description, and I'll predict its genre!")

# Text input
user_input = st.text_area("Enter movie description:")

if st.button("Predict Genre"):
    if user_input.strip():
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        st.success(f"**Predicted Genre:** 🎭 {prediction}")
    else:
        st.warning("Please enter a valid movie description!")
