import os
import glob
import json
import pandas as pd
import numpy as np

#base_dir = os.path.join('./data')
base_dir = os.path.join('./temp')
train_dir = base_dir

fn1 = base_dir+'/trip_utter1.csv'
fn2 = base_dir+'/trip_conv1.csv'

df3 = pd.read_csv(fn1, sep='\t')
df4 = pd.read_csv(fn2, sep='\t')

word_to_id = {'none':0, 'yes':1, 'no':2}
id_to_word = {0:'none', 1:'yes', 2:'no'}

def update_vocab(text):
    text = text.replace('.', ' .')
    words = text.split(' ');
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    return len(words)

def load_data():
    questions = list(df3['text'].values)
    answers = list(df3['eval'].values) 


    new_qs = []
    new_as = []
    max = 0

    for i in range(len(answers)): 
        a = answers[i].replace('[','').replace(']','') 
        t_a = [item.strip(" '") for item in a.split(",")] 
        if (t_a[0] == 'eval'):
            new_qs.append(questions[i]) 
            new_as.append(t_a) 

    for q in new_qs:
        wcnt = update_vocab(q)
        if (max < wcnt):
            max = wcnt 

    rows = len(new_qs) 

    corpus = np.array([], dtype=np.int32)
    for i, text in enumerate(new_qs):
        text = text.replace('.', ' .')
        words = text.split(' ');
        corpus = np.append(corpus, np.array([word_to_id[w] for w in words]))
        pad = max - len(words)
        corpus = np.append(corpus, np.zeros(pad)) 

    x_corpus = corpus.reshape(len(new_qs), max) 

    corpus = np.array([], dtype=np.int32)
    for i, text in enumerate(new_as):
        words=text[1:]
        corpus = np.append(corpus, np.array([word_to_id[w] for w in words])) 

    t_corpus = corpus.reshape(len(new_qs), 9)

    x_corpus = x_corpus.astype(np.int64)
    t_corpus = t_corpus.astype(np.int64)

    split_at = len(x_corpus) - len(x_corpus) // 10
    (x_train, x_test) = x_corpus[:split_at], x_corpus[split_at:]
    (t_train, t_test) = t_corpus[:split_at], t_corpus[split_at:]

    return (x_train, t_train), (x_test, t_test)

def get_vocab():
    return word_to_id, id_to_word
