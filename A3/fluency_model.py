from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, AutoModelForSequenceClassification
import torch
import pandas as pd
import json
# from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from functions import *

def trainer_fluency(file_train, file_val):
    # read train.jsonl file and dev.jsonl
    train_data = pd.read_json(file_train, lines=True) 
    train_data = train_data.sample(frac=1).reset_index(drop=True)

    BATCH_SIZE = 8
    MAX_LEN_INPUT = 128
    MAX_LEN_OUTPUT = 8

    input_texts = []
    max_len = 0
    for i in (range(len(train_data))):
        entry = train_data.iloc[i]
        input_t = ''
        input_t += entry['input']
        max_len = max(max_len,len(input_t.split()))
        input_texts.append(input_t)

    target1_train = []
    for i in (range(len(train_data))):
        entry = train_data.iloc[i]
        output_t = ''
        if (entry['pattern'] == ''):
            output_t += 'FLUENT'
        else:
            output_t += 'DISFLUENT'
        target1_train.append(output_t)

    train_dict = {'input_text' : input_texts, 'target_text' : target1_train}

    tokenizer = AutoTokenizer.from_pretrained('t5-small',model_max_length = 512)
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')

    input_tokenized = tokenizer(input_texts,truncation = True,padding='max_length', max_length = MAX_LEN_INPUT, return_tensors = 'pt')
    target_tokenized = tokenizer(target1_train, truncation = True, padding = 'max_length', max_length = MAX_LEN_OUTPUT, return_tensors = 'pt')
        
    train_dataset = TensorDataset(input_tokenized,target_tokenized)
    train_data_loader = DataLoader(train_dataset,batch_size = BATCH_SIZE, shuffle = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCHS = 8
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay =0.01)

    model = model.to(device)

    for epochs in range(EPOCHS):
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
            epoch_loss += loss.item()
            # progress_bar.update(1)
        # print("Epoch {} Loss {}".format(epochs,epoch_loss/len(train_data_loader)))
    model.save_pretrained("./t5_fluency_model")



    val_data = pd.read_json(file_val,lines = True)
    val_texts = []
    max_len = 0
    for i in (range(len(val_data))):
        entry = val_data.iloc[i]
        input_t = ''
        input_t += entry['input']
        max_len = max(max_len,len(input_t.split()))
        val_texts.append(input_t)

    val_tokenized = tokenizer(val_texts,truncation = True,padding='max_length', max_length = MAX_LEN_INPUT, return_tensors = 'pt')

    val_dataset = ValDataset(val_tokenized)

    val_data_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)

    model.eval()
    generated_test_texts = []
    with torch.no_grad():
        # progress_bar = tqdm(range(len(val_data_loader)), position = 0, leave = True)
        for batch in val_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            prediction = model.generate(input_ids,attention_mask=attention_mask,max_new_tokens = MAX_LEN_OUTPUT, pad_token_id = tokenizer.pad_token_id)
            prediction_decoded = tokenizer.batch_decode(prediction, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            generated_test_texts.extend(prediction_decoded)
            # progress_bar.update(1)
            
    with open("./val_fluency.txt","w") as f:
        for i in range(len(generated_test_texts)):
            f.write("{} # {} \n".format(val_texts[i],generated_test_texts[i]))


        