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

glove_dim = 300

# labels_dic = {
#     'O': 0,
#     'Chemical_Compound' : 1,
#     'Biological_Molecule' : 2,
#     'Species' : 3
# }

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

sentences = read_data('data/train.txt')
sentences_val = read_data('data/dev.txt')
word2idx, embedding_matrix = get_vocab(sentences, glove_dim)
input_batches, gold_batches = batch_data(sentences, word2idx, labels_dic)

input_train, gold_train = batch_data(sentences, word2idx, labels_dic, 1)
input_val, gold_val = batch_data(sentences_val, word2idx, labels_dic, 1)

# model = LSTM(glove_dim, 512, embedding_matrix, len(labels_dic.keys()), 0.5)
model = torch.load('check4_f.pt', map_location=torch.device('cpu'))
model.eval()

for i in tqdm(range(len(input_val))):
    gold_val[i] = gold_val[i].flatten()
    # flattened_input_val[i] = input_val[i].flatten()
gold_val = torch.cat(gold_val)

validate_step(model, input_val, gold_val)