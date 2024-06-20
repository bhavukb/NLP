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

dic = {
    'Biological_Molecule' : 0,
    'Chemical_Compound' : 1,
    'Species' : 2
}

chunks = {
    'O': 0,
    'S': 1,
    'B': 2,
    'I': 3,
    'E': 4,
}


# Load the data
sentences = read_data('data/train.txt')
sentences_val = read_data('data/dev.txt')
word2idx, embedding_matrix = get_vocab(sentences, glove_dim)
input_batches, gold_batches = batch_data(sentences, word2idx, chunks, 32, 2)

alph = fasttext_alph(sentences)
pwords, plabels = sentence_to_phase2(sentences, alph, dic)
pwords_val, plabels_val = sentence_to_phase2(sentences_val, alph, dic)

input_train, gold_train = batch_data(sentences, word2idx, chunks, 1, 2)
input_val, gold_val = batch_data(sentences_val, word2idx, chunks, 1, 2)

# Train the model
# lstm_model = train_model(input_batches, gold_batches, input_val, gold_val, embedding_matrix, output_dim = len(chunks.keys()), input_dim = glove_dim, hidden_dim = 128, num_epochs=10)
ft_model = train_stage_two(pwords, plabels, alph, len(dic.keys()), 200, 128, 10)

# Test the model
def evaluate(y_true, y_pred):
    f1_micro = metrics.f1_score(y_true, y_pred, average="micro", labels=[1,2,3,4,5,6,7,8,9,10,11,12])
    f1_macro = metrics.f1_score(y_true, y_pred, average="macro", labels=[1,2,3,4,5,6,7,8,9,10,11,12])
    return ( f1_micro, f1_macro, (f1_micro+f1_macro) / 2 )

def convert(chunk, label):
    if (chunk == 0):
        return 0
    return chunk * 3 - 2 + label

# input_val, gold_val = batch_data(sentences_val, word2idx, labels_dic, 1)
# for i in tqdm(range(len(input_val))):
#     gold_val[i] = gold_val[i].flatten()
#     # flattened_input_val[i] = input_val[i].flatten()
# gold_val = torch.cat(gold_val)

# sentences_val.sort(key = lambda x:len(x))
# validation_output = []
# for i in tqdm(range(len(input_val))):
#     val = lstm_model(input_val[i])[0]
#     val = torch.argmax(val, dim=1)
#     for j in range(len(val)):
#         if val[j] != 0:
#             # print(sentences[i][j][0])
#             idx = ft_to_idx(sentences_val[i][j][0], alph)
#             label = ft_model(idx)
#             label = torch.argmax(label)
#             val[j] = convert(val[j], label)

#     validation_output.append(val)
# validation_output = torch.cat(validation_output)

# f1_score_micro_val, f1_score_macro_val, f1_score_val = evaluate(gold_val, validation_output)
# print("Validation Micro: ", f1_score_micro_val)
# print("Validation Macro: ", f1_score_macro_val)
# print("Validation F1 Score: ", f1_score_val)





validation_output = []
for inp in tqdm(pwords_val):
    output_tensor = ft_model(inp)
    output = torch.argmax(output_tensor)
    validation_output.append(output)

f1_score_micro_val, f1_score_macro_val, f1_score_val = evaluate(plabels_val, validation_output)
print("Validation Micro: ", f1_score_micro_val)
print("Validation Macro: ", f1_score_macro_val)
print("Validation F1 Score: ", f1_score_val)