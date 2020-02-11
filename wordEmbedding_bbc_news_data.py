# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:51:00 2020

@author: sabab
"""

import re, string
import pandas as pd
from collections import defaultdict
import spacy
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
STOPWORDS = set(stopwords.words('english'))



def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]'% re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    
    if len(text)>2:
        return ' '.join(word for word in text.split() if word not in STOPWORDS)

def lemmatizer(text):
    global count
    nlp = spacy.load('en_core_web_sm', disable = ['ner', 'parser']) # disabling named entity recognition
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    count+=1
    print("Progress : {}".format(count))
    return " ".join(sent)

def tsne_plot(model):
    "Create TSNE model and plot it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(18, 18)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

def w2VMOdel(minFreq, window, featureVector, workers):
    w2v_model = Word2Vec(min_count = 400,
                     window = 5,
                     size = 200,
                     workers = 4)
    return w2v_model


#count = 0
    
minFreq = 200 # freq of the words to be considered for the model
window = 5 #window for the model
featureVector = 100 # dimension of the vector 
workers = 4

fileName = "CleanDataFrame"
#df = pd.read_csv('bbc-text.csv')
#df_clean = pd.DataFrame(df.text.apply(lambda x: clean_text(x)))
#df_clean["text_lemmatize"] = df_clean.apply(lambda x:lemmatizer(x['text']), axis = 1)
#df_clean["text_lemmatize_clean"] = df_clean["text_lemmatize"].str.replace('-PRON-','')
#df_clean.to_pickle(fileName)

#load clean data
df_clean = pd.read_pickle(fileName)

listOfWords = [row.split() for row in df_clean['text_lemmatize_clean']]

word_freq = defaultdict(int)

for document in listOfWords:
    for word in document:
        word_freq[word]+=1
        
#print(sorted(word_freq, key = word_freq.get, reverse = True)[:10])
        
w2v_model = w2VMOdel(minFreq, window, featureVector, workers)

w2v_model.build_vocab(listOfWords)
w2v_model.train(listOfWords, total_examples = w2v_model.corpus_count, epochs = w2v_model.iter)
w2v_model.init_sims(replace=True)

word2 = w2v_model.wv.most_similar(positive=['tv'])
word3 = w2v_model.wv.similarity('tv', 'home')

tsne_plot(w2v_model)

