import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import sklearn.metrics as metrics
from sklearn.utils import class_weight
import numpy as np
import pandas as pd

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_matrix, num_classes, dp):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = False)
        # self.embeddings2 = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        # self.pc = nn.Linear(input_dim, 212, num_classes)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional = True, dropout = dp)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional = True)
        # self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True, bidirectional = True)
        # self.lstm2 = nn.LSTM(2 * 128, 128, batch_first=True, bidirectional = True)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
        
    def forward(self, x):
        # print("X: " + str(x.shape))
        embeds = self.embedding(x)
        # output = self.pc(embeds)
        # output1, (h_1, c_1) = self.lstm1(embeds)
        output, (h_2, c_2) = self.lstm(embeds)
        # print("OUTPUT SHAPE: " + str(output.shape))
        tag_space = self.fc(output)
        # print("TAG SPACE SHAPE: " + str(tag_space.shape))
        tag_scores = nn.functional.log_softmax(tag_space, dim=2)
        # print("TAG SCORES SHAPE: " + str(tag_scores.shape))
        return tag_scores


def train_model(input_batches, output_batches, input_val, gold_val, embedding_matrix, output_dim, input_dim = 100, hidden_dim = 256, num_epochs=25, dp = 0.5):
    lstm_model = LSTM(input_dim, hidden_dim, embedding_matrix, output_dim, dp)
    y0 = []
    for i in range(len(output_batches)):
        y0.append(output_batches[i].flatten())
    y0 = torch.cat(y0)
    y0 = y0.numpy()
    class_weights = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                        y = y0
                                    )
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    print(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    # flattened_input_val = []
    for i in tqdm(range(len(input_val))):
        gold_val[i] = gold_val[i].flatten()
        # flattened_input_val[i] = input_val[i].flatten()
    gold_val = torch.cat(gold_val)
    

    best_model = None
    best_score = 0
    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch + 1))
        for i in tqdm(range(len(input_batches))):
            input_batch = input_batches[i]
            output_batch = output_batches[i]
            output_batch = torch.flatten(output_batch, 0, -1)
            # print("OUTPUT BATCH SHAPE: " + str(output_batch.shape))
            output_tensor = lstm_model(input_batch)
            output_tensor = torch.flatten(output_tensor, 0, 1)
            # print("OUTPUT TENSOR SHAPE: " + str(output_tensor.shape))
            loss = criterion(output_tensor, output_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        score = validate_step(lstm_model, input_val, gold_val)
        if score > best_score:
            best_score = score
            best_model = lstm_model


    return lstm_model, best_model

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



# 2 models
# First model categorizes O B S I E
# Second model applies a fast-text like system and categorizes the sentence into correct labels

class LSTM2(nn.Module):
    def __init__(self, alph_dim, embedding_dim, hidden_dim, num_classes):
        super(LSTM2, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(alph_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional = True, dropout = 0.6)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
        
    def forward(self, x):
        x = torch.tensor(x)
        embeds = self.embedding(x)
        output, (h_n, c_n) = self.lstm(embeds)
        sum_lasthidden = torch.cat([output[-1, :self.hidden_dim], output[-1, self.hidden_dim:]])
        tag_space = self.fc(sum_lasthidden)
        tag_scores = nn.functional.log_softmax(tag_space, dim=0)
        return tag_scores
        
def train_stage_two(input, gold, alph, output_dim, embedding_dim = 100, hidden_dim = 20, num_epochs=25):
    model = LSTM2(len(alph), embedding_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch + 1))
        outputs = []
        for i in tqdm(range(len(input))):
            inp = input[i]
            # print("Golde: " + str(gold[i]))
            out = np.array([gold[i]])
            out = torch.tensor(out)
            output_tensor = model(inp)
            # print("OUTPUT TENSOR SHAPE: " + str(output_tensor.shape))
            # print("OUT SHAPE: " + str(out.shape))

            # loss = criterion(output_tensor.reshape(1, output_dim), out.reshape(1,1))
            loss = criterion(torch.reshape(output_tensor, (1, output_dim)), out)
            output_tensor = torch.argmax(output_tensor, dim=0)
            outputs.append(output_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        f1_score_micro_val, f1_score_macro_val, f1_score_val = evaluate(gold, outputs)
        print("Validation Micro: ", f1_score_micro_val)
        print("Validation Macro: ", f1_score_macro_val)
        print("Validation F1 Score: ", f1_score_val)

    return model