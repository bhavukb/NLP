from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
import json
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from functions import *

def trainer_function(file_train, file_val):
    train_data = pd.read_json(file_train, lines=True) 
    val_data = pd.read_json(file_val,lines = True)
    train_data = train_data.sample(frac=1).reset_index(drop=True)

    BATCH_SIZE = 4
    MAX_LEN_INPUT = 512
    MAX_LEN_OUTPUT = 128

    my_file = open("./val_fluency.txt", "r")
    data = my_file.read()
    val_fluency = data.split("\n")

    input_texts = []
    max_len = 0
    for i in (range(len(train_data))):
        entry = train_data.iloc[i]
        input_t = ''
        input_t += entry['input']
        input_t += process_fluency_train(entry['pattern'])
        input_t += ' ' + entry['input']
        input_t += ' #HISTORY# ' + process_history(entry['history'])
        input_t += ' #USER_LISTS# ' + process_user_lists(entry['user_lists'])
        input_t += ' #USER_NOTES# ' + process_user_notes(entry['user_notes'])
        input_t += ' #USER_CONTACTS# ' + process_contacts(entry['user_contacts'])
        max_len = max(max_len,len(input_t.split()))
        input_texts.append(input_t)

    val_texts = []
    max_len = 0
    for i in (range(len(val_data))):
        entry = val_data.iloc[i]
        input_t = ''
        input_t += entry['input']
        input_t += process_fluency_val(val_fluency[i])
        input_t += ' ' + entry['input']
        input_t += ' #HISTORY# ' + process_history(entry['history'])
        input_t += ' #USER_LISTS# ' + process_user_lists(entry['user_lists'])
        input_t += ' #USER_NOTES# ' + process_user_notes(entry['user_notes'])
        input_t += ' #USER_CONTACTS# ' + process_contacts(entry['user_contacts'])
        max_len = max(max_len,len(input_t.split()))
        val_texts.append(input_t)

    target_texts = []
    max_len = 0
    for i in (range(len(train_data))):
        entry = train_data.iloc[i]
        output_t = ''
        output_t += entry['output']
        target_texts.append(output_t)

    val_targets = []
    for i in (range(len(val_data))):
        entry = val_data.iloc[i]
        output_t = ''
        output_t += entry['output']
        val_targets.append(output_t)

    # prepare train and val dictionaries
    train_dict = {'input_text' : input_texts, 'target_text' : target_texts}
    val_dict = {'input_text' : val_texts, 'target_text' : val_targets}

    # combine train and val dict
    combined_dict = {}
    for key in train_dict.keys():
        combined_dict[key] = train_dict[key] + val_dict[key]
    train_dict = combined_dict

    # tokenize input and target texts
    tokenizer = AutoTokenizer.from_pretrained('t5-base', model_max_length = 512)
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
    input_tokenized = tokenizer(train_dict["input_text"],truncation = True,padding='max_length', max_length = MAX_LEN_INPUT, return_tensors = 'pt')
    target_tokenized = tokenizer(train_dict["target_text"], truncation = True, padding = 'max_length', max_length = MAX_LEN_OUTPUT, return_tensors = 'pt')


    train_dataset = TensorDataset(input_tokenized,target_tokenized)
    train_data_loader = DataLoader(train_dataset,batch_size = BATCH_SIZE, shuffle = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCHS = 9
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay =0.01)
    model = model.to(device)

    for epochs in range(EPOCHS):
        # print("Training Epoch: " + str(epochs))
        epoch_loss = 0 
        # progress_bar = tqdm(range(len(train_data_loader)), position = 0, leave = True)
        model.train()    
        for batch in train_data_loader:
            optimizer.zero_grad()    
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            predict = model(input_ids,attention_mask = attention_mask,labels = labels)
            loss = predict.loss
            loss.backward()
            optimizer.step()
            epoch_loss += predict.loss.item()
            # progress_bar.update(1)

    model.save_pretrained("mt1190727_mt1190683_model")