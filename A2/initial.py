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

TRAIN_PATH = sys.argv[1]
VAL_PATH = sys.argv[2]

# Load the data
# sentences = read_data('data/train.txt')
# sentences_val = read_data('data/dev.txt')
sentences = read_data(TRAIN_PATH)
sentences_val = read_data(VAL_PATH)
word2idx, embedding_matrix = get_vocab(sentences, glove_dim)
input_batches, gold_batches = batch_data(sentences, word2idx, labels_dic)

input_train, gold_train = batch_data(sentences, word2idx, labels_dic, 1)
input_val, gold_val = batch_data(sentences_val, word2idx, labels_dic, 1)

# Train the model
lstm_model, best_model = train_model(input_batches, gold_batches, input_val, gold_val, embedding_matrix, output_dim = len(labels_dic.keys()), input_dim = glove_dim, hidden_dim = 512, num_epochs=30, dp=0.5)



# training_output = []
# for i in tqdm(range(len(input_train))):
#     training_output.append(lstm_model(input_train[i]))
# validation_output = []
# val_tensor = torch.tensor([])
# for i in tqdm(range(len(input_val))):
#     val = lstm_model(input_val[i])[0]
#     val = torch.argmax(val, dim=1)
#     validation_output.append(val)
#     gold_val[i] = gold_val[i].flatten()
#     input_val[i] = input_val[i].flatten()

# validation_output = torch.cat(validation_output)
# gold_val = torch.cat(gold_val)


# Evaluate the model by finding precision and recall and then print the scores
def get_f1_score(input, gold, output):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(gold)):
        if output[i] == 0:
            if gold[i] != 0:
                fn += 1
            else:
                tn += 1
        else:
            if gold[i] == output[i]:
                tp += 1
            else:
                fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    # f1_score = f1_score / len(output)
    return precision, recall

def evaluate(y_true, y_pred):
    f1_micro = metrics.f1_score(y_true, y_pred, average="micro", labels=[1,2,3,4,5,6,7,8,9,10,11,12])
    f1_macro = metrics.f1_score(y_true, y_pred, average="macro", labels=[1,2,3,4,5,6,7,8,9,10,11,12])
    return ( f1_micro, f1_macro, (f1_micro+f1_macro) / 2 )

# f1_score_micro_train, f1_score_macro_train, f1_score_train = evaluate(gold_train, training_output)
# precision_train, recall_train = get_f1_score(gold_train, training_output)

# f1_score_micro_val, f1_score_macro_val, f1_score_val = evaluate(gold_val, validation_output)
# precision_val, recall_val = get_f1_score(input_val, gold_val, validation_output)

# print("Training Precision: ", precision_train)
# print("Training Recall: ", recall_train)
# print("Training Micro: ", f1_score_micro_train)
# print("Training Macro: ", f1_score_macro_train)
# print("Training F1 Score: ", f1_score_train)


# print("Validation Precision: ", precision_val)
# print("Validation Recall: ", recall_val)
# print("Validation Micro: ", f1_score_micro_val)
# print("Validation Macro: ", f1_score_macro_val)
# print("Validation F1 Score: ", f1_score_val)


pickle.dump([word2idx], open("./extras.pkl", 'wb'))
torch.save(best_model, "./mt1190683_model")