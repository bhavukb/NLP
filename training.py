import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

def evaluate(y_true, y_pred):
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    # return (f1_micro+f1_macro)/2
    return (f1_micro, f1_macro, (f1_micro+f1_macro) / 2 )

def split(df, k):
    df_split = np.array_split(df, k)
    return df_split

def upsample(df, repeats = 1, stars_upto = 3, freq = None):
    if freq is None:
        to_repeat = (df['class'] <= stars_upto)
        df_to_concat = [df[to_repeat]] * repeats
        return pd.concat([df] + df_to_concat, ignore_index=True)
    else:
        lst = []
        for i in range(5):
            current = df[df['class'] == i]
            df_to_concat = [current] * freq[i]
            lst += df_to_concat
        return pd.concat(lst, ignore_index=True)
    
def vectorize(df_train, df_test, vectorizer):
    print("Starting Vectorization...")
    training_features = vectorizer.fit_transform(df_train['text'])
    training_features = training_features.toarray()
    testing_features = vectorizer.transform(df_test['text'])
    testing_features = testing_features.toarray()
    print("Vectorization Complete")
    return (training_features, testing_features)

def simple(df_train, df_test, i):
    # vectorizer = CountVectorizer(ngram_range = (1, 3), tokenizer = None, preprocessor = None, stop_words = None, max_features = 3000)
    vectorizer = TfidfVectorizer(ngram_range = (1, 3), tokenizer = None, preprocessor = None, stop_words = None, max_features = 3000)
    training_features, testing_features = vectorize(df_train, df_test, vectorizer)
    print("Training fold ", i, "...")
    classifier = LinearSVC(verbose=True, C=1.0, max_iter=1000)
    classifier.fit(training_features, df_train['class'])
    y_pred = classifier.predict(testing_features)
    return y_pred

def multinb(df_train, df_test, i):
    vectorizer = TfidfVectorizer(ngram_range = (1, 3), tokenizer = None, preprocessor = None, stop_words = None, max_features = 3000)
    training_features, testing_features = vectorize(df_train, df_test, vectorizer)
    print("Training fold ", i, "...")
    classifier = MultinomialNB(fit_prior = True, alpha = 1.0)
    classifier.fit(training_features, df_train['class'])
    y_pred = classifier.predict(testing_features)
    return y_pred

def bernoullinb(df_train, df_test, i):
    vectorizer = TfidfVectorizer(ngram_range = (1, 3), tokenizer = None, preprocessor = None, stop_words = None, max_features = 3000)
    training_features, testing_features = vectorize(df_train, df_test, vectorizer)
    print("Training fold ", i, "...")
    classifier = BernoulliNB(fit_prior = True, alpha = 1.0)
    classifier.fit(training_features, df_train['class'])
    y_pred = classifier.predict(testing_features)
    return y_pred

def two_layer(df_train, df_test, i):
    print("Training fold ", i, "...")
    # vectorizer = CountVectorizer(ngram_range = (1, 3), tokenizer = None, preprocessor = None, stop_words = None, max_features = 3000)
    vectorizer = TfidfVectorizer(ngram_range = (1, 3), tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    training_features, testing_features = vectorize(df_train, df_test, vectorizer)
    y = df_train['class']

    training_features_low = []
    y_low = []
    for i in range(len(y)):
        if y[i] <= 3:
            training_features_low.append(training_features[i])
            y_low.append(y[i])
            y[i] = 0
    training_features = np.array(training_features)

    classifier = LinearSVC(verbose=True, C=1.0, max_iter=1000)
    classifier.fit(training_features, y)
    classifier2 = LinearSVC(verbose=True, C=1.0, max_iter=1000)
    classifier2.fit(training_features_low, y_low)

    y_pred = classifier.predict(testing_features)
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            y_pred[i] = classifier2.predict(testing_features[i].reshape(1, -1))

    return y_pred



def onevsrest(df_train, df_test, i):
    print("Training fold ", i, "...")
    vectorizer = TfidfVectorizer(ngram_range = (1, 3), tokenizer = None, preprocessor = None, stop_words = None, max_features = 3000)
    training_features, testing_features = vectorize(df_train, df_test, vectorizer)
    y = df_train['class']

    classifier = OneVsRestClassifier(LinearSVC(verbose=True, C=1.0, max_iter=1000))
    classifier.fit(training_features, y)
    y_pred = classifier.predict(testing_features)
    return y_pred

def onevsone(df_train, df_test, i):
    print("Training fold ", i, "...")
    vectorizer = TfidfVectorizer(ngram_range = (1, 3), tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    training_features, testing_features = vectorize(df_train, df_test, vectorizer)
    y = df_train['class']

    classifier = OneVsOneClassifier(LinearSVC(verbose=True, C=1.0, max_iter=1000))
    # classifier = OneVsOneClassifier(BernoulliNB(fit_prior = True, alpha = 1.0))
    classifier.fit(training_features, y)
    y_pred = classifier.predict(testing_features)
    return y_pred

def seperated(df_train, df_test, i):
    print("Training fold ", i, "...")
    # vectorizer = CountVectorizer(ngram_range = (1, 3), tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    vectorizer = TfidfVectorizer(ngram_range = (1, 3), tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
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

    classifier = LinearSVC(verbose=True, C=1.0, max_iter=1000)
    classifier.fit(training_features, y)
    classifier2 = LinearSVC(verbose=True, C=1.0, max_iter=1000)
    classifier2.fit(training_features_low, y_low)
    classifier3 = LinearSVC(verbose=True, C=1.0, max_iter=1000)
    classifier3.fit(training_features_high, y_high)

    y_pred = classifier.predict(testing_features)

    # y_test_temp = list(df_test['class'])
    # for i in range(len(y_test_temp)):
    #     if y_test_temp[i] <= 3:
    #         y_test_temp[i] = 0
    #     else:
    #         y_test_temp[i] = 1
    # fig, axis = plt.subplots(figsize=(10,7))
    # ConfusionMatrixDisplay.from_predictions(y_test_temp, y_pred, ax=axis)
    # _ = axis.set_title(f"Confusion Matrix for iteration {i}")
    # plt.show()

    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            y_pred[i] = classifier2.predict(testing_features[i].reshape(1, -1))
        elif y_pred[i] == 1:
            y_pred[i] = classifier3.predict(testing_features[i].reshape(1, -1))

    return y_pred


def new_simple(df_train, df_test, i):
    print("Training fold ", i, "...")
    vectorizer = TfidfVectorizer(ngram_range = (1, 3), tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    training_features, testing_features = vectorize(df_train, df_test, vectorizer)
    y = df_train['class']

    classifier = LinearSVC(verbose=True, C=1.0, max_iter=1000, class_weight = {1: 3, 2: 3, 3: 2, 4: 1, 5: 1})
    classifier.fit(training_features, y)
    y_pred = classifier.predict(testing_features)
    return y_pred

def new_two_layer(df_train, df_test, i):
    print("Training fold ", i, "...")
    # vectorizer = CountVectorizer(ngram_range = (1, 3), tokenizer = None, preprocessor = None, stop_words = None, max_features = 3000)
    vectorizer = TfidfVectorizer(ngram_range = (1, 3), tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    training_features, testing_features = vectorize(df_train, df_test, vectorizer)
    y = df_train['class']

    training_features_low = []
    y_low = []
    for i in range(len(y)):
        if y[i] <= 3:
            training_features_low.append(training_features[i])
            y_low.append(y[i])
            y[i] = 0
    training_features = np.array(training_features)

    classifier = LinearSVC(verbose=True, C=1.0, max_iter=1000)
    classifier.fit(training_features, y)
    classifier2 = LinearSVC(verbose=True, C=1.0, max_iter=1000)
    classifier2.fit(training_features_low, y_low)

    y_pred = classifier.predict(testing_features)
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            y_pred[i] = classifier2.predict(testing_features[i].reshape(1, -1))

    return y_pred

def logreg(df_train, df_test, i):
    print("Training fold ", i, "...")
    # vectorizer = CountVectorizer(ngram_range = (1, 3), tokenizer = None, preprocessor = None, stop_words = None, max_features = 3000)
    vectorizer = TfidfVectorizer(ngram_range = (1, 2), tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    training_features, testing_features = vectorize(df_train, df_test, vectorizer)
    y = df_train['class']

    classifier = LogisticRegression(verbose=1, random_state=20, max_iter=100, penalty='l2', C=1.0, solver='saga', multi_class='multinomial', dual=False)
    classifier.fit(training_features, y)
    y_pred = classifier.predict(testing_features)
    return y_pred

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
        # y_pred = simple(df_train, df_test, i)
        # y_pred = two_layer(df_train, df_test, i)
        # y_pred = multinb(df_train, df_test, i)
        # y_pred = bernoullinb(df_train, df_test, i)
        # y_pred = onevsrest(df_train, df_test, i)
        # y_pred = onevsone(df_train, df_test, i)
        # y_pred = new_simple(df_train, df_test, i)
        # y_pred = new_two_layer(df_train, df_test, i)
        # y_pred = seperated(df_train, df_test, i)
        y_pred = logreg(df_train, df_test, i)

        
        
        # Evaluate the results
        f1_micro, f1_macro, f1 = evaluate(y_true, y_pred)
        result[i] = (f1_micro, f1_macro, f1)
        print("Fold", i, ":", result[i])

    for i in range(k):
        print("Fold", i, ":", result[i])


# # example df
# df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'b': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
# kfolds = KFold(n_splits=5, shuffle=True, random_state=17)
# # print i th fold
# for i, (train_index, test_index) in enumerate(kfolds.split(df)):
#     print("Fold", i)
#     print(df.iloc[train_index])
#     print(df.iloc[test_index])