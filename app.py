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

for dependency in ("brown","names","wordnet","average_perceptron_tagger","universal_tagset"):
    nltk.download(dependency)
import warnings
warnings.filterwarnings("ignore")
np.random.seed(123)

data=pd.read_csv("/home/user/Desktop/DatasetBank/labeledTrainData.tsv",sep='\t')

print(data.head())
print(data.shape)
print(data.isnull().sum())

stop_words=stopwords.words("english")
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


# review column data get cleaned here
data['cleaned_review']=data['review'].apply(text_cleaning)
print(data["cleaned_review"][:5])
print(data.head())

X=data["cleaned_review"]
y=data.sentiment.values

# split data into train and validate

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=y,
)

print(X_train.shape,"-----------",y_train.shape)

# Create a classifier in pipeline
sentiment_classifier = Pipeline(steps=[
                                 ('pre_processing',TfidfVectorizer(lowercase=False)),
                                 ('naive_bayes',MultinomialNB())
                                 ])

sentiment_classifier.fit(X_train,y_train)
y_pred=sentiment_classifier.predict(X_valid)
print("accuracy score is",accuracy_score(y_pred,y_valid))


import joblib
joblib.dump(sentiment_classifier,'review_classification_model.pkl')









