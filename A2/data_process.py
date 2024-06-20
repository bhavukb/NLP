from torch.utils.data import DataLoader, Dataset
import torch
import torchtext.vocab as vocab
import numpy as np

def read_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    sentences = []
    current_sentence = []
    for line in lines:
        line = line.strip()
        if not line:
            sentences.append(current_sentence)
            current_sentence = []
        else:
            word, label = line.split('\t')
            current_sentence.append([word, label])
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

def get_vocab(sentences, glove_dim = 200):
    words = []
    for s in sentences:
        for w, _ in s:
            words.append(w)
    words = list(set(words))
    word2idx = {w: i for i, w in enumerate(words)}

    # Get a matrix of word vectors
    glove = vocab.GloVe(name='6B', dim=glove_dim)
    matrix_len = len(word2idx)
    embedding_matrix = torch.zeros((matrix_len, glove_dim))
    for word, idx in word2idx.items():
        if word in glove.stoi:
            embedding_matrix[idx] = torch.tensor(glove.vectors[glove.stoi[word]])
    # Add unk
    embedding_matrix = torch.cat((embedding_matrix, torch.zeros(1, glove_dim)))
    word2idx['<unk>'] = matrix_len

    return word2idx, embedding_matrix

def sentence_to_tensor(sentence, word2idx, labels_dic, type = 1):
    tensor = []
    output = []
    for w, l in sentence:
        if w not in word2idx:
            tensor.append(word2idx['<unk>'])
        else:
            tensor.append(word2idx[w])
        
        if (type == 1):
            output.append(labels_dic[l])
        else:
            output.append(labels_dic[l[0]])

        # if (l == 'O'):
        #     output.append(0)
        # else:
        #     l = l[2:]
        #     output.append(labels_dic[l])
    return torch.tensor(tensor), torch.tensor(output)

def batch_data(sentences, word2idx, labels_dic, size=32, type = 1):
    sentences = sorted(sentences, key=lambda x: len(x))
    batch_size = size
    input_batches = []
    output_batches = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        max_length = max([len(s) for s in batch])
        input_batch, output_batch = [], []
        for s in batch:
            padded_s = s + [['.', 'O']] * (max_length - len(s))
            input_tensor, output_tensor = sentence_to_tensor(padded_s, word2idx, labels_dic, type)
            input_batch.append(input_tensor)
            output_batch.append(output_tensor)
        input_batches.append(torch.stack(input_batch))
        output_batches.append(torch.stack(output_batch))
        # input_batches.append(input_batch)
        # output_batches.append(output_batch)
    
    return input_batches, output_batches
    
    return input_batches, output_batches

def fasttext_alph(sentences):
    alph = {}
    idx = 0
    for sentence in sentences:
        for word, label in sentence:
            if label == 'O':
                continue
            word = "~~" + word + "~~"
            for i in range(2, len(word)):
                if (word[i-2:i+1] not in alph):
                    alph[ word[i-2:i+1] ] = idx
                    idx += 1
    alph['~~~'] = idx
    return alph

def ft_to_idx(word, alph):
    word = "~~" + word + "~~"
    idx = []
    for i in range(2, len(word)):
        if (word[i-2:i+1] not in alph):
            idx.append(alph['~~~'])
        else:
            idx.append(alph[word[i-2:i+1]])
    return idx

def sentence_to_phase2(sentences, alph, dic):
    pwords, plabels = [], []
    for sentence in sentences:
        for word, label in sentence:
            if label == 'O':
                continue
            pwords.append(ft_to_idx(word, alph))
            plabels.append(dic[label[2:]])
    return pwords, plabels

def read_test(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    sentences = []
    current_sentence = []
    for line in lines:
        line = line.strip()
        if not line:
            sentences.append(current_sentence)
            current_sentence = []
        else:
            word= line
            current_sentence.append(word)
    if current_sentence:
        sentences.append(current_sentence)
    return sentences


def sentence_to_tensor_test(sentence, word2idx, labels_dic, type = 1):
    tensor = []
    for w in sentence:
        if w not in word2idx:
            tensor.append(word2idx['<unk>'])
        else:
            tensor.append(word2idx[w])
    return torch.tensor(tensor)

def batch_test(sentences, word2idx, labels_dic, size=32, type = 1):
    # sentences = sorted(sentences, key=lambda x: len(x))
    batch_size = size
    input_batches = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        max_length = max([len(s) for s in batch])
        input_batch = []
        for s in batch:
            padded_s = s + ['.'] * (max_length - len(s))
            input_tensor = sentence_to_tensor_test(padded_s, word2idx, labels_dic, type)
            input_batch.append(input_tensor)
        input_batches.append(torch.stack(input_batch))
    
    return input_batches