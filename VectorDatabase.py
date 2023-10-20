import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from PIL import Image
from io import BytesIO
from nltk.tokenize import RegexpTokenizer
import nltk
from gensim.models import Word2Vec
import gensim.utils
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("data.csv")

def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)

def make_lower_case(text):
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

df['cleaned'] = df['Desc'].apply(_removeNonAscii)
df['cleaned'] = df.cleaned.apply(make_lower_case)
df['cleaned'] = df.cleaned.apply(remove_stop_words)
df['cleaned'] = df.cleaned.apply(remove_punctuation)
df['cleaned'] = df.cleaned.apply(remove_html)

df['cleaned'].replace('', np.nan, inplace=True)
df = df[df['cleaned'].notna()]

corpus = []
for words in df['cleaned']:
    corpus.append(words.split())


word2vec_model = Word2Vec(vector_size=300, window=5, min_count=2, workers=-1)
temporary_file = "temporary_model.bin"
with open(temporary_file, 'wb') as f_out, open("GoogleNews-vectors-negative300.bin", 'rb') as f_in:
    header = gensim.utils.to_utf8(f"{len(word2vec_model.wv.index_to_key)} {word2vec_model.wv.vector_size}\n")
    f_out.write(header)
    f_out.write(f_in.read())
word2vec_model.wv = gensim.models.KeyedVectors.load_word2vec_format(temporary_file, binary=True)
word2vec_model.build_vocab(corpus)
word2vec_model.train(corpus, total_examples=word2vec_model.corpus_count, epochs=15) 

import os
os.remove(temporary_file)

def get_document_vectors(document, word2vec_model):
    document_embedding_list = []
    for doc in document:
        document_embedding = np.zeros(word2vec_model.vector_size)
        valid_words = 0
        for word in doc:
            if word in word2vec_model.wv.key_to_index:
                word_index = word2vec_model.wv.key_to_index[word]
                word_embedding = word2vec_model.wv.vectors[word_index]
                document_embedding += word_embedding
                valid_words += 1

        if valid_words > 0:
            document_embedding /= valid_words

        document_embedding_list.append(document_embedding)

    return document_embedding_list

document_embedding_list = get_document_vectors(df['cleaned'], word2vec_model)
cosine_similarities = cosine_similarity(document_embedding_list, document_embedding_list)

def recommendations(title):
    books = df[['title', 'image_link']]

    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = indices[title]

    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]

    book_indices = [i[0] for i in sim_scores]

    recommend = books.iloc[book_indices].reset_index(drop=True)

    book_titles = []
    
    for index, row in recommend.iterrows():
        book_titles.append(row['title'])

    return book_titles

recommended_books = recommendations("The Da Vinci Code")
print(recommended_books)
