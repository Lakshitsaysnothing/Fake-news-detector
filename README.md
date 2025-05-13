# 📰 Fake News Detector

This is a simple Python-based machine learning project that classifies news articles as either fake or real using the Multinomial Naive Bayes algorithm and TF-IDF vectorization.

- `fake_news_predictor.py` — trains the model and saves `model.pkl` & `vectorizer.pkl`
- `predict_from_saved_model.py` — loads saved model and lets you input custom news text for prediction
- `model.pkl` & `vectorizer.pkl` — saved trained components


# How to run locally 
# Step 01: Clone the repository:
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector


# Step02: Install dependencies
Make sure python is installed(version 3.6+). Then install dependecies:

pip install -r requirement.txt
This install dependencies like scikit.learn, pandas, numpy etc.

# Step03: Run the prediction script

python predict_from_saved_model.py


## Example Input/Output :

Enter news text (or type 'exit' to quit)

The Prime Minister announced a new scheme to support farmers.

Prediction: Real