
import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load saved model and vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load processed dataset
df = pd.read_csv("processed_reviews.csv")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# TF-IDF matrix for recommendations
review_matrix = tfidf.transform(df["CleanReview"])

st.title("Amazon Product Review Analytics")

review = st.text_area("Enter Product Review")

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned_review = clean_text(review)
        vector = tfidf.transform([cleaned_review])

        prediction = model.predict(vector)
        sentiment = label_encoder.inverse_transform(prediction)[0]

        st.success(f"Predicted Sentiment: {sentiment}")

        similarity = cosine_similarity(vector, review_matrix)
        top3 = similarity.argsort()[0][-3:][::-1]

        st.subheader("Top 3 Similar Products")

        for idx in top3:
            st.write("**Review Title**", df.iloc[idx]["ReviewTitle"])
            st.write("**Country:**", df.iloc[idx]["Country"])
            st.write("**Rating:**", df.iloc[idx]["Rating"])
            st.write(df.iloc[idx]["ReviewText"])
            st.write("---")
