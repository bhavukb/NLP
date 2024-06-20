import pandas as pd
import numpy as np
import contractions
import string
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import nltk
from tqdm import tqdm
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Take in a list of paragraphs and return the list after removing contractions
def remove_contractions(paragraphs_list):
    new_paragraphs_list = []
    for paragraph in paragraphs_list:
        new_paragraphs_list.append(contractions.fix(paragraph))
    return new_paragraphs_list

# Lemmatize the list of paragraphs
def lemmatize(paragraphs_list):
    new_paragraphs_list = []
    lemmatizer = WordNetLemmatizer()
    for paragraph in tqdm(paragraphs_list):
        tagged = pos_tag(word_tokenize(paragraph))
        sentence = ""
        for word, tag in tagged:
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
            if not wntag:
                lemma = word
            else:
                lemma = lemmatizer.lemmatize(word, wntag)
            sentence += lemma + " "
        new_paragraphs_list.append(sentence)
    return new_paragraphs_list

stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'of', 'at', 'by', 'for', 'with', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'both', 'each', 'few', 'some', 'such', 'own', 'same', 'so', 's', 't', 'can', 'will', 'just', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'can', 'could', 'will', 'would', 'shall', 'should', 'has', 'have', 'had', 'may', 'might', 'be'])

neg_words = set(['no', 'nor', 'not'])

# Remove stop words from the list of paragraphs
def remove_stop_words(paragraphs_list):
    new_paragraphs_list = []
    for paragraph in paragraphs_list:
        sentence = ""
        for word in paragraph.split():
            if word not in stop_words:
                sentence += word + " "
        new_paragraphs_list.append(sentence)
    return new_paragraphs_list

# Remove dashes
def remove_dashes(paragraphs_list):
    new_paragraphs_list = []
    for paragraph in paragraphs_list:
        temp = paragraph.replace("-", " ")
        temp = temp.replace("/", " ")
        new_paragraphs_list.append(temp)
    return new_paragraphs_list

# Negate the next k words after a negative
def negate(paragraphs_list, k = 3):
    new_paragraphs_list = []
    for paragraph in paragraphs_list:
        sentence = ""
        i = 0
        for word in paragraph.split():
            if word in neg_words:
                i = 3
            elif word in string.punctuation:
                sentence += word + " "
            elif i > 0:
                sentence += "not_" + word + " "
                i -= 1
            else:
                sentence += word + " "
        new_paragraphs_list.append(sentence)
    return new_paragraphs_list