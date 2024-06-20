import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pickle
import sys

from pre_processing import remove_contractions
from pre_processing import lemmatize
from pre_processing import remove_stop_words
from pre_processing import remove_dashes
from pre_processing import negate
from data_handling import data_augment
from final_separated import model_train
from training import upsample, evaluate


INPUT_PATH  = sys.argv[1]
train_df = pd.read_csv(INPUT_PATH, header=None)
train_df[0] = train_df[0].values.astype('U')


print("Starting Preprocessing...")
paragraph_list = list(train_df[0])
paragraph_list = remove_contractions(paragraph_list)
# paragraph_list = lemmatize(paragraph_list)
paragraph_list = remove_stop_words(paragraph_list)
paragraph_list = remove_stop_words(paragraph_list)
paragraph_list = remove_dashes(paragraph_list)
paragraph_list = negate(paragraph_list, 3)
print("Preprocessing Complete")

# with open("preprocessed", "wb") as fp:   #Pickling
#     pickle.dump(paragraph_list, fp)

# with open("preprocessed", "rb") as fp:   # Unpickling
#     paragraph_list = pickle.load(fp)
# print(paragraph_list[0:10])

print("Starting Training...")
# cross_val(paragraph_list, train_df[1], 5)

df_train = pd.DataFrame()
df_train['text'] = paragraph_list
df_train['class'] = train_df[1]
df_train = upsample(df_train, 1, 3)

(classifier, classifier2, classifier3, vectorizer) = model_train(df_train)

try:
    MODEL_PATH = sys.argv[2]
except:
    MODEL_PATH = '2019MT10683.model'

# save the models and tfidf vectorizer
pickle.dump([classifier, classifier2, classifier3, vectorizer], open(MODEL_PATH, 'wb'))

print("Training Complete")

