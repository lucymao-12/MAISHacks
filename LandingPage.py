from cProfile import label
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
from flask import request, render_template
from sklearn.naive_bayes import MultinomialNB


#nltk.download()

def lists():

    data_test=pd.read_csv("Corona_NLP_test.csv", encoding ='latin1')
    data_train=pd.read_csv("Corona_NLP_train.csv" ,encoding ='latin1')


    #print(data_train.columns)
    #data_train

    features_og = data_train.iloc[:, 4].values
    labels = data_train.iloc[:, 5].values
    
    #print(labels[31000:])


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
    
    #print(len(features_train))
    features_train = features_train[31000:]
    labels = labels[31000:]

    #print(type(features_train))
    
    

    features = vectorizer.fit_transform(features_train).toarray()
    #new_feature=vectorizer.transform(new_value).toarray()

    #for i in features:
        #print(i)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

    #test = X_test[0]

    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    text_classifier.fit(X_train, y_train)
    #text_classifier.

    #print(type(text_classifier))

    pred = text_classifier.predict(X_test)

    #print(pred[-1])

    new_value="this is an informative tweet nothing to see here just some words no emotions"
    features_train.append(new_value)
    np.append(labels, "Negative")
    features = vectorizer.fit_transform(features_train).toarray()
    print(text_classifier.predict([features[-1]]))
    #print(confusion_matrix(y_test, pred))
    #print(classification_report(y_test, pred))
    print(accuracy_score(y_test, pred))

    return text_classifier

lists()

'''
    for i in range(len(labels)):
        if labels[i] == "Neutral":
            labels[i] = 0
        elif labels[i] == "Positive":
            labels[i] = 1
        elif labels[i] == "Extremely Positive":
            labels[i] = 2
        elif labels[i] == "Negative":
            labels[i] = -1
        else:
            labels[i] = -2

    for i in labels:
        print(i)
'''
    #return (features_train, labels)

    
    #print(pred)

'''
    #for i in features:
        #print(i)



app = Flask(__name__)

@app.route("/")
def in_text():
    return render_template("site.html")

@app.route('/', )
def site():
'''
    