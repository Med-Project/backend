import re
import json
import gensim
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.spatial import distance

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from gensim.models.word2vec import Word2Vec

from SiameseNetwork import SiameseNetwork

###############################################

words = ['мм', 'рт', 'ст', 'г', 'м', 'мл', 'кг', 'мг', 'см', 'ч', 'е', 'х', 'a', 'd', 'b', 'мц', 'лг', 'бт',
        'млн', 'фсг'] 
set_stopWords = stopwords.words("russian") + words
set_stopWords = set(set_stopWords)

model = Word2Vec.load('./w2v/med_w2v')
stemmer = SnowballStemmer("russian")

X_topic = []
Y_topic = []

X_path = 'topic_vectors.txt'
Y_path = 'topic_names.txt'

X_topic = np.loadtxt(X_path)
with open(Y_path, 'r') as f:
    for line in f.readlines():
        Y_topic.append(line.replace('\n',''))
Y_topic = np.array(Y_topic)

bolezni = pd.read_csv('bolezni.csv')
X_categ = []
for i in bolezni['topics']:
    toks = i.split(' $ ')
    X_categ.append(toks[0])
X_categ = np.array(X_categ)

name2topic = {}
for i in range(X_topic.shape[0]):
    name2topic[str(Y_topic[i])] = str(X_categ[i])


def deleteStopWords(text):
    stop = ['др. греч.', 'греч.', 'лат.', 'др.', 'и т. д', 'и т.д', 'т. е.', 'т.е.', 'т.н.', 'т. н.', 'т.п.', 'т. п.',
                'т. д', 'т.д', 'мм. рт. ст.', 'мм рт. ст.', 'мм.рт.ст', 'МЕДПОИСК.РУ', 'Контактная форма ниже', 'ºС']
    for i in stop:
        text = text.replace(i, '')
    return text

def clearText(text):
    text = deleteStopWords(text)
    text = text.replace('\xad', '')
    text = text.replace('\n\n', '.')
    text = text.replace('..', '.')
    text = text.replace(':', ' ')
    text = text.replace(';', '.')
    text = text.replace('?', '.')
    text = text.replace('!', '.')
    text = re.sub(r'\[[\d+]*\]', '', text)
    text = re.sub(r'\([\d+]*\)', '', text)
    text = re.sub('\d+', ' ', text)
    text = re.sub('[^\w.]', ' ', text)
    
    final_s = ''
    for sen in text.split('.'):
        sen = sen.strip()
        tmp_t = ''
        for tok in sen.split(' '):
            tok = tok.strip()
            if(tok.lower() not in set_stopWords):
                tmp_t += tok + ' '
        final_s += tmp_t + '.'
    return final_s

def getVector(s):
    tmp = clearText(s)
    tokens = gensim.utils.simple_preprocess(tmp)
    tmp2 = [word for word in tokens if word not in stopwords.words("russian")]
    word_vectors = []
    for tok in tmp2:
        s = stemmer.stem(tok)
        if(s in model.wv.vocab):
            word_vectors.append(model[s])
    word_vectors = np.array(word_vectors)
    final_vec = np.mean(word_vectors, axis=0)
    return final_vec

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def getSim_w2v(vec, cnt):
    sim = []
    for i in X_topic:
        sim.append(cos_sim(vec, i))
    indexs = np.argsort(np.array(sim))
    
    res = [(Y_topic[i]) for i in indexs[-cnt:]]
    return list(reversed(res))

def getSim_siamese(vector, all_vectors, cnt):
    dist = []
    for i in range(all_vectors.shape[0]):
        dist.append(np.linalg.norm(vector-all_vectors[i]))
    dist = np.array(dist)

    indexs = np.argsort(np.array(dist))
    res = [(name2topic[Y_topic[i]], Y_topic[i]) for i in indexs[:cnt]]
    return res

def prediction(s, siamese):
    text_vector = getVector(s)
    if(np.isnan(text_vector).any()):
        return -1
    
    res = getSim_w2v(text_vector, 15)

    all_embed = siamese.o1.eval({siamese.x1: X_topic})
    text_embed = siamese.o1.eval({siamese.x1: text_vector.reshape([-1, 250])})
    siamese_res = getSim_siamese(text_embed, all_embed, 15)
    return (res, siamese_res)

