from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
import torch
import pandas as pd
import json
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader


def process_history(history):
    string = ''
    for old_entry in history:
        string += json.dumps(old_entry) + ' '
    return string

def process_user_lists(user_lists):
    string = ''
    for old_entry in user_lists:
        string += json.dumps(old_entry) + ' '
    return string

def process_user_notes(user_notes):
    string = ''
    for notes in user_notes:
        string += json.dumps(notes) + ' '
    return string


def process_contacts(contacts):
    return ' '.join(contacts)

def process_fluency_val(line):
    word = line.split()[-1]
    if (word == 'DISFLUENT'):
        return ' #DISFLUENT# '
    else:
        return ' #FLUENT# '
    
def process_fluency_train(word):
    if (word == 'disfluency'):
        return ' #DISFLUENT# '
    else:
        return ' #FLUENT# '
    
class TensorDataset(Dataset):
    def __init__(self, input_tokenized, target_tokenized):
        self.input_ids = input_tokenized['input_ids']
        self.attention_mask = input_tokenized['attention_mask']
        self.labels = target_tokenized['input_ids']
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self,i):
        d = {}
        d['input_ids'] = self.input_ids[i]
        d['attention_mask'] = self.attention_mask[i]
        d['labels'] = self.labels[i]
        return d
    
class ValDataset(Dataset):
    def __init__(self, input_tokenized):
        self.input_ids = input_tokenized['input_ids']
        self.attention_mask = input_tokenized['attention_mask']
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self,i):
        d = {}
        d['input_ids'] = self.input_ids[i]
        d['attention_mask'] = self.attention_mask[i]
        return d

