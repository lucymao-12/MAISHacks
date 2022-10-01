from flask import Flask
import pandas as pd
import numpy as np
import sklearn as sk
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#nltk.download()
data_test=pd.read_csv("Corona_NLP_test.csv", encoding ='latin1')
data_train=pd.read_csv("Corona_NLP_train.csv" ,encoding ='latin1')


print(data_train.columns)
#data_train

features_og = data_train.iloc[:, 4].values
labels = data_train.iloc[:, 5].values


features_train = []
for sentence in range(len(features_og)):
    feature_train=re.sub(r'\W', ' ', str(features_og[sentence]))
    feature_train=re.sub(r'\^[a-zA-Z]\s+', ' ', feature_train)
    feature_train=re.sub(r'\s+[a-zA-Z]\s+', ' ', feature_train)
    feature_train=re.sub(r'\s+', ' ', feature_train, flags=re.I)
    feature_train=re.sub(r'^b\s+', '', feature_train)

    feature_train=feature_train.lower()
    features_train.append(feature_train)

vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
features = vectorizer.fit_transform(features_train).toarray()



X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

pred = text_classifier.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
#print(pred)




#for i in features:
    #print(i)

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, world</p>"
