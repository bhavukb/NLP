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

def vectorize(df_train, df_test, vectorizer):
    print("Starting Vectorization...")
    training_features = vectorizer.fit_transform(df_train['text'])
    training_features = training_features.toarray()
    testing_features = vectorizer.transform(df_test['text'])
    testing_features = testing_features.toarray()
    print("Vectorization Complete")
    return (training_features, testing_features)

def first_separation(training_features, y, testing_features):
    classifier = LinearSVC(verbose=True, C=1.0, max_iter=1000)
    # classifier = LogisticRegression(verbose=1, random_state=20, max_iter=100, penalty='l2', C=1.0, solver='saga')
    # classifier = SVC(verbose=True, C=1.0, max_iter=1000)
    classifier.fit(training_features, y)
    y_pred = classifier.predict(testing_features)
    return y_pred
    
def second_separation(training_features_low, y_low, training_features_high, y_high, y_pred, testing_features):
    classifier2 = LinearSVC(verbose=True, C=1.0, max_iter=500)
    # classifier2 = LogisticRegression(verbose=1, random_state=20, max_iter=100, penalty='l2', C=1.0, solver='saga')
    classifier2.fit(training_features_low, y_low)
    # classifier3 = LinearSVC(verbose=True, C=1.0, max_iter=1000)
    classifier3 = LogisticRegression(verbose=1, random_state=20, max_iter=100, penalty='l2', C=1.0, solver='saga')
    classifier3.fit(training_features_high, y_high)

    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            y_pred[i] = classifier2.predict(testing_features[i].reshape(1, -1))
        elif y_pred[i] == 1:
            y_pred[i] = classifier3.predict(testing_features[i].reshape(1, -1))
    return y_pred
    
def separation_control(df_train, df_test, i):
    print("Training fold ", i, "...")
    vectorizer = CountVectorizer(ngram_range = (1, 2), tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    # vectorizer = TfidfVectorizer(ngram_range = (1, 2), tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    training_features, testing_features = vectorize(df_train, df_test, vectorizer)
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

    y_first = first_separation(training_features, y, testing_features)
    y_second = second_separation(training_features_low, y_low, training_features_high, y_high, y_first, testing_features)

    return y_second

    
def cross_val(paragraph_list, y, k):
    print("Begin Cross Validation")
    df = pd.DataFrame()
    df['text'] = paragraph_list
    df['class'] = y
    kfolds = KFold(n_splits=k, shuffle=True, random_state=17)
    result = {}
    print("Splitting complete")

    for i, (train_index, test_index) in enumerate(kfolds.split(df)):
        print("Starting Fold", i)
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        df_train = upsample(df_train, 1, 3)
        # df_train = upsample(df_train, freq = [3, 2, 1, 1, 1])

        # Run the training technique
        y_true = df_test['class']
        y_pred = separation_control(df_train, df_test, i)

        # Evaluate the results
        f1_micro, f1_macro, f1 = evaluate(y_true, y_pred)
        result[i] = (f1_micro, f1_macro, f1)
        print("Fold", i, ":", result[i])

    for i in range(k):
        print("Fold", i, ":", result[i])