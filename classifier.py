import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle

# Ensure required NLTK datasets are downloaded
nltk.download("punkt")
nltk.download("stopwords")

# Sample dataset to reduce training time
train_data = pd.read_csv("train_data.txt", sep=":::", names=["TITLE", "GENRE", "DESCRIPTION"], engine="python")
train_sample = train_data.sample(5000, random_state=42)  # Take 5k random samples

# Data cleaning function
def clean_text(text):
    stemmer = LancasterStemmer()
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = nltk.word_tokenize(text)
    text = " ".join([stemmer.stem(word) for word in words if word not in stop_words and len(word) > 2])
    return text

# Apply cleaning
train_sample["TextCleaning"] = train_sample["DESCRIPTION"].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_sample["TextCleaning"])
y_train = train_sample["GENRE"]

# Train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Save model and vectorizer
with open("svm_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

# Load pre-trained model and vectorizer
with open("svm_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit UI
st.title("🎬 Movie Genre Predictor")
st.write("Enter a movie description, and I'll predict its genre!")

# Text input
user_input = st.text_area("Enter movie description:")

if st.button("Predict Genre"):
    if user_input:
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        st.success(f"Predicted Genre: **{prediction}** 🎭")
    else:
        st.warning("Please enter a movie description!")
