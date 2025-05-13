import pickle
import string
#load the model  and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

#Clean text funtion (same as before)
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return text
#Predict function
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    return "Real" if prediction[0] == 1 else "Fake"

# Interactive loop
while True:
    user_input = input("\nEnter news text (or type 'exit' to quit):\n")
    if user_input.lower() == 'exit':
        break
    print("Prediction:", predict_news(user_input))
