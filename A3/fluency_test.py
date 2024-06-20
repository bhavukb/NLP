from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
import json
# from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from functions import *

def tester_fluency(file_test, output):
    test_data = pd.read_json(file_test, lines = True)

    BATCH_SIZE = 8
    MAX_LEN_INPUT = 512
    MAX_LEN_OUTPUT = 128

    test_texts = []
    max_len = 0
    for i in range(len(test_data)):
        entry = test_data.iloc[i]
        input_t = ''
        input_t += entry['input']
        max_len = max(max_len,len(input_t.split()))
        test_texts.append(input_t)

    tokenizer = AutoTokenizer.from_pretrained('t5-small',model_max_length = 512)
    model = AutoModelForSeq2SeqLM.from_pretrained('./t5_fluency_model')

    test_tokenized = tokenizer(test_texts,truncation = True,padding='max_length', max_length = MAX_LEN_INPUT, return_tensors = 'pt')
    test_dataset = ValDataset(test_tokenized)
    test_data_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    generated_test_texts = []
    with torch.no_grad():
        # progress_bar = tqdm(range(len(test_data_loader)), position = 0, leave = True)
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            prediction = model.generate(input_ids,attention_mask=attention_mask,max_new_tokens = MAX_LEN_OUTPUT, pad_token_id = tokenizer.pad_token_id)
            prediction_decoded = tokenizer.batch_decode(prediction, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            generated_test_texts.extend(prediction_decoded)
            # progress_bar.update(1)

    with open("./test_fluency_preds.txt","w") as f:
        for i in range(len(generated_test_texts)):
            f.write("{} # {} \n".format(test_texts[i],generated_test_texts[i]))