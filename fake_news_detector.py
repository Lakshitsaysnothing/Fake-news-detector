import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake['label'] = 0
true['label'] = 1
data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)

# Preprocessing without NLTK
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

data['cleaned'] = data['text'].apply(clean_text)

# TF-IDF Vectorizer with built-in English stopwords
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(data['cleaned']).toarray()
y = data['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict custom news
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    return "Real" if prediction[0] == 1 else "Fake"

# Test example
print(predict_news("The government passed a new bill today that ..."))
#saving the model and vectorizer in the disk
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)