
import streamlit as st

# Basic packages
import numpy as np
import pandas as pd

# Scikit Learn Packages
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from string import punctuation
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib

for dependency in ("brown","names","wordnet","average_perceptron_tagger","universal_tagset"):
    nltk.download(dependency)
import warnings
warnings.filterwarnings("ignore")
np.random.seed(123)


stop_words=stopwords.words("english")

@st.cache_data
def text_cleaning(text,remove_stop_words=True,lemmatize_words=True):
    # Remove punctuations and numbers from the data
    text=re.sub(r"[^A-Za-z0-9]"," ",text)
    text=re.sub(r"\'s"," ",text)
    text =  re.sub(r'http\S+',' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) # remove numbers


     # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer() 
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    
    # Return a list of words
    return(text)


# now we use saved model to predict

def make_prediction(review):
    cleaned_review=text_cleaning(review)
    cleaned_review=[cleaned_review]
    model=joblib.load("/home/user/Desktop/STREAMLIT_PROJECT/review_classification_model.pkl")
    print("1 executed")
    predicted_review=model.predict(cleaned_review)
    print("2 executed")
    print("3 executed")
    print(predicted_review)
    return predicted_review

# obj=make_prediction("good its actually too good and nice")
# print("obj is ",obj)

# Set the app title
st.title("Sentiment Analyisis App")
st.write(
    "A simple machine laerning app to predict the sentiment of a movie's review"
)

# Declare a form to receive a movie's review
form = st.form(key="my_form")
review = form.text_input(label="Enter the text of your movie review")
submit = form.form_submit_button(label="Make Prediction")

if submit:
    # make prediction from the input text
    result = make_prediction(review)
 
    # Display results of the NLP task
    st.header("Results")
 
    if int(result) == 1:
        st.write("This is a positive review")
    else:
        st.write("This is a negative review")






