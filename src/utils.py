import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import re
import pickle

from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder


def load_data(file_path):
    return pd.read_csv(file_path, encoding='utf-8', sep=';')


def plot_top_stopwords_barchart(text, stopwords, n=10, figsize=(10, 5)):
    stopwords = set(stopwords)

    new = text.str.split()
    new = new.values.tolist()
    all_words = [word for i in new for word in i]
    dic = defaultdict(int)
    for word in all_words:
        if word in stopwords:
            dic[word] += 1

    top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:n]
    x, y = zip(*top)
    plt.figure(figsize=figsize)
    plt.bar(x, y)


def plot_word_distribution_over_intents(dataframe, word):
    dataframe['intent'][dataframe['question']\
        .str.contains(word)].value_counts().\
        plot(kind='bar', align='center', width=0.5,
             figsize=(10, 5))


def clean_sentence(sentence,
                   stopwords,
                   clean_policy,
                   lemmatizer):
    # removed every punctuation and special characters.
    clean = re.sub(r'[{}]'.format(clean_policy), " ", sentence)

    # tokenize
    tokens = word_tokenize(clean)

    # remove stopwords
    tokens = filter(lambda x: x not in stopwords, tokens)

    # lemmatize
    return [lemmatizer.lemmatize(i.lower()) for i in tokens]


def clean_sentences(sentences,
                    stopwords,
                    clean_policy,
                    lemmatizer):
    words = []
    for s in sentences:
        clean = clean_sentence(s,
                               stopwords,
                               clean_policy,
                               lemmatizer)
        words.append(clean)
    return words


def create_tokenizer(words, filters):
    """Creates a tokenizer objects that encodes all the questions by
    mapping each token to an integer"""
    token = Tokenizer(filters=filters)
    token.fit_on_texts(words)
    return token


def load_tokenizer(file_path):
    with open(file_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def save_tokenizer(tokenizer, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_words_max_length(words):
    """Get the length of the maximum length word"""
    return len(max(words, key=len))


def encode_sentences(token, words):
    return token.texts_to_sequences(words)


def encode_output_labels(token, labels):
    output = encode_sentences(token, labels)
    return np.array(output).reshape(len(output), 1)


def pad_sentences(encoded_sentences, max_length):
    return pad_sequences(encoded_sentences, maxlen=max_length, padding='post')


def one_hot(x):
    o = OneHotEncoder(sparse=False)
    return o.fit_transform(x)