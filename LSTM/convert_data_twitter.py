# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:25:52 2018

@author: Xiaomi
"""

import pandas as pd
import numpy as np
import Stemmer
import re
from collections import OrderedDict

from keras.preprocessing import sequence

stemmer = Stemmer.Stemmer('russian')

def stem(data):
#    data = re.sub("[^\w]", " ", data).split()
    data = re.findall("[а-яА-Я]+", data)
    return stemmer.stemWords(data)

negative = pd.read_csv('twitter/negative.csv', sep=';')
positive = pd.read_csv('twitter/positive.csv', sep=';')

c_n = positive['ttext'].map(stem)
c_p = negative['ttext'].map(stem)
#
#c_n = c_n[:15000]
#c_p = c_p[:15000]

all_words = []
for i in c_n:
    for word in i:
        all_words.append(word)
for j in c_p:
    for word in j:
        all_words.append(word)
        
uniq_words = list(OrderedDict.fromkeys(all_words))

c_p1 = []
for row in c_p:
    lst = []
    for word in row:
        lst.append(uniq_words.index(word))
    c_p1.append(lst)

c_n1 = []
for row in c_n:
    lst = []
    for word in row:
        lst.append(uniq_words.index(word))
    c_n1.append(lst)
#    
df_pos = pd.DataFrame(c_p1)
df_neg = pd.DataFrame(c_n1)
df_pos.to_csv('positive_set.csv',sep=';')
df_neg.to_csv('negative_set.csv',sep=';')
