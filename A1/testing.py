import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import sys
import pickle

from pre_processing import remove_contractions
from pre_processing import lemmatize
from pre_processing import remove_stop_words
from pre_processing import remove_dashes
from pre_processing import negate
from data_handling import data_augment
from final_separated import model_train
from training import upsample, evaluate


TEST_PATH   = sys.argv[2]
try:
    MODEL_PATH = sys.argv[1]
except:
    MODEL_PATH = '2019MT10683.model'
OUTPUT_PATH = sys.argv[3]

classifier, classifier2, classifier3, vectorizer = pickle.load(open(MODEL_PATH,'rb'))


test_df = pd.read_csv(TEST_PATH, header=None)
test_df[0] = test_df[0].values.astype('U')

print("Starting Preprocessing...")
paragraph_list = list(test_df[0])
paragraph_list = remove_contractions(paragraph_list)
# paragraph_list = lemmatize(paragraph_list)
paragraph_list = remove_stop_words(paragraph_list)
paragraph_list = remove_stop_words(paragraph_list)
paragraph_list = remove_dashes(paragraph_list)
paragraph_list = negate(paragraph_list, 3)
print("Preprocessing Complete")

testing_features = vectorizer.transform(paragraph_list)

y_first = classifier.predict(testing_features)
for i in range(len(y_first)):
    if y_first[i] == 0:
        y_first[i] = classifier2.predict(testing_features[i].reshape(1, -1))
    elif y_first[i] == 1:
        y_first[i] = classifier3.predict(testing_features[i].reshape(1, -1))

#make pandas dataframe
y_first = pd.DataFrame(y_first)
y_first.to_csv(OUTPUT_PATH, index=False, header=False)
