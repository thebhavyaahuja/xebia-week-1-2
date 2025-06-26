import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
# importing Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

#load the dataset
df = pd.read_csv("IMDB Dataset.csv")

# mapping the sentiment
df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# clean the text
def clean_text(text):
  text = re.sub(r"[^a-zA-Z]"," ",text).lower()
  tokens = text.split()
  tokens = [word for word in tokens if word not in stop_words]
  return " ".join(tokens)

df["cleaned_review"]=df["review"].apply(clean_text)

df["cleaned_review"].head()

# feature extraction
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_review"])
y=df["sentiment"]
# divide into train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# model training
model = MultinomialNB()
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)
#calculate accuracy
# print all metrics
print("accuracy is:")
print(accuracy_score(y_test,y_pred))
print("Precision is:")
print(precision_score(y_test,y_pred))
print("Recall is:")
print(recall_score(y_test,y_pred))
print("F1 score is:")
print(f1_score(y_test,y_pred))
print("Confusion matrix is:")
print(confusion_matrix(y_test,y_pred))
print("***************  Classification Report: ***************")
print(classification_report(y_test,y_pred))

# Save the model and vectorizer
import joblib
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')   