import pandas as pd
import numpy as np
import nltk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from data_process import *
from model import *
from sklearn.metrics import precision_recall_fscore_support
import sklearn.metrics as metrics
import pickle
import sys

TEST_PATH = sys.argv[1]
OUT_PATH = sys.argv[2]

glove_dim = 300

labels_dic = {
    'O': 0,
    'S-Biological_Molecule': 1,
    'S-Chemical_Compound': 2,
    'S-Species': 3,
    'B-Biological_Molecule': 4,
    'B-Chemical_Compound': 5,
    'B-Species': 6,
    'I-Biological_Molecule': 7,
    'I-Chemical_Compound': 8,
    'I-Species': 9,
    'E-Biological_Molecule': 10,
    'E-Chemical_Compound': 11,
    'E-Species': 12
}

inverse_dic = {
    0: 'O',
    1: 'S-Biological_Molecule',
    2: 'S-Chemical_Compound',
    3: 'S-Species',
    4: 'B-Biological_Molecule',
    5: 'B-Chemical_Compound',
    6: 'B-Species',
    7: 'I-Biological_Molecule',
    8: 'I-Chemical_Compound',
    9: 'I-Species',
    10: 'E-Biological_Molecule',
    11: 'E-Chemical_Compound',
    12: 'E-Species'
}

model = torch.load('./mt1190683_model', map_location=torch.device('cpu'))
model.eval()
[word2idx] = pickle.load(open('./extras.pkl', 'rb'))

# sentences_val = read_test('test1.txt')
sentences_val = read_test(TEST_PATH)
input_val = batch_test(sentences_val, word2idx, labels_dic, 1)

# for i in tqdm(range(len(input_val))):
#     gold_val[i] = gold_val[i].flatten()
#     # flattened_input_val[i] = input_val[i].flatten()
# gold_val = torch.cat(gold_val)

def validate_step(model, input_val, gold_val):
    validation_output = []
    for i in tqdm(range(len(input_val))):
        val = model(input_val[i])[0]
        val = torch.argmax(val, dim=1)
        validation_output.append(val)
    validation_output = torch.cat(validation_output)
    f1_score_micro_val, f1_score_macro_val, f1_score_val = evaluate(gold_val, validation_output)
    print("Validation Micro: ", f1_score_micro_val)
    print("Validation Macro: ", f1_score_macro_val)
    print("Validation F1 Score: ", f1_score_val)
    return f1_score_val


def evaluate(y_true, y_pred):
    f1_micro = metrics.f1_score(y_true, y_pred, average="micro", labels=[1,2,3,4,5,6,7,8,9,10,11,12])
    f1_macro = metrics.f1_score(y_true, y_pred, average="macro", labels=[1,2,3,4,5,6,7,8,9,10,11,12])
    return ( f1_micro, f1_macro, (f1_micro+f1_macro) / 2 )



# f = open("outputfile.txt", "w")
f = open(OUT_PATH, "w")

for i in tqdm(range(len(input_val))):
    val = model(input_val[i])[0]
    val = torch.argmax(val, dim=1)
    for j in range(len(val)):
        k = inverse_dic[val[j].item()]
        f.write(k + "\n")
    f.write("\n")

f.close()
