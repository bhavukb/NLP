import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from training import upsample, evaluate

def vectorize_f(df_train, vectorizer):
    print("Starting Vectorization...")
    training_features = vectorizer.fit_transform(df_train['text'])
    training_features = training_features.toarray()
    print("Vectorization Complete")
    return (training_features, vectorizer)

def model_train(df_train):
    print("Training fold...")
    vectorizer = TfidfVectorizer(ngram_range = (1, 2), tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    training_features, vectorizer = vectorize_f(df_train, vectorizer)
    y = df_train['class']

    training_features_low = []
    y_low = []
    training_features_high = []
    y_high = []
    for i in range(len(y)):
        if y[i] <= 3:
            training_features_low.append(training_features[i])
            y_low.append(y[i])
            y[i] = 0
        else:
            training_features_high.append(training_features[i])
            y_high.append(y[i])
            y[i] = 1
    training_features = np.array(training_features)

    classifier = LinearSVC(verbose=True, C=1.0, max_iter=1000)
    classifier2 = LinearSVC(verbose=True, C=1.0, max_iter=1000)
    classifier3 = LogisticRegression(verbose=1, random_state=20, max_iter=100, penalty='l2', C=1.0, solver='saga')

    classifier.fit(training_features, y)
    classifier2.fit(training_features_low, y_low)
    classifier3.fit(training_features_high, y_high)

    return (classifier, classifier2, classifier3, vectorizer)

