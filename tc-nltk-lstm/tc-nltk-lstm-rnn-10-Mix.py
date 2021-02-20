#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Text Classification / tc-nltk-lstm-rnn-10-mix.ipynb
# Gourav Siddhad
# 27-Mar-2019


# In[21]:

print('Importing Libraries', end='')

from itertools import cycle
from scipy import interp
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical, plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.optimizers import RMSprop
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.models import Model, load_model
import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MultiLabelBinarizer, minmax_scale
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from nltk.corpus import reuters, stopwords
from nltk import word_tokenize
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
import re
from numpy.random import seed
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

print(' - Done')


# In[3]:


documents = reuters.fileids()
print('Total Documents -', len(documents))

print('Extracting (Id, Docs and Labels)', end='')
train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))

train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]
all_docs = train_docs
all_docs += test_docs

train_labels = [reuters.categories(doc_id) for doc_id in train_docs_id]
test_labels = [reuters.categories(doc_id) for doc_id in test_docs_id]
all_labels = train_labels
all_labels += test_labels
print(' - Done')

del train_docs
del test_docs
del train_labels
del test_labels

print('Documents - ', len(all_docs))
print('Labels  - ', len(all_labels))

# List of categories
categories = reuters.categories()
print('Categories - ', len(categories))


# In[4]:


print('Caching Stop Words', end='')
cachedStopWords = stopwords.words("english")
print(' - Done')


def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(
        token) and len(token) >= min_length, tokens))
    return filtered_tokens


# In[5]:


print('Total Articles - ', len(all_docs))
allwords = set()
i = 0
for doc in all_docs:
    if i % 100 is 0:
        print(i, end=' ')
    doc = tokenize(doc)
    for word in doc:
        allwords.add(word)
    i += 1


# In[6]:


print('All Words - ', len(allwords))
# print(allwords)


# In[7]:


print('Sorting Train:Test Docs', end='')
X_train, X_test, y_train, y_test = train_test_split(
    all_docs, all_labels, test_size=0.2, random_state=42)
print(' - Done')

maxwords = len(allwords)

print('Tokenizing', end='')
tk = Tokenizer(num_words=maxwords)
tk.fit_on_texts(X_train)
tk.fit_on_texts(X_test)
index_list_train = tk.texts_to_sequences(X_train)
index_list_test = tk.texts_to_sequences(X_test)
print(' - Done')


# In[8]:


# max of index_list_train
# max of index_list_test

maxlen = 200

for i in index_list_train:
    if len(i) > maxlen:
        maxlen = len(i)
print(maxlen)

for i in index_list_test:
    if len(i) > maxlen:
        maxlen = len(i)
print(maxlen)


# In[9]:


# maxlen = 1600
print('MaxLen - ', maxlen)
print('Padding Sequences', end='')
x_train = sequence.pad_sequences(index_list_train, maxlen=maxlen)
x_test = sequence.pad_sequences(index_list_test, maxlen=maxlen)
print(' - Done')

print('Binarizing MultiLabels', end='')
lb = MultiLabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)
print(' - Done')


# In[10]:


del all_docs
del all_labels


# In[13]:


config = tf.ConfigProto(device_count={"CPU": 8})
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


# In[14]:


def RNN():
    with tf.device('/gpu:0'):
        inputs = Input(name='inputs', shape=[maxlen])
        layer1 = Embedding(maxwords, 1024)(inputs)
    with tf.device('/cpu:0'):
        layer2 = LSTM(256)(layer1)
        layer3 = Dense(128)(layer2)
    with tf.device('/gpu:0'):
        layer4 = Activation('relu')(layer3)
        layer5 = Dropout(rate=0.1)(layer4)  # rate = 1-keep_prob, keep_prob=0.5
        layer6 = Dense(len(categories))(layer5)
        layer7 = Activation('softmax')(layer6)
    model = Model(inputs=inputs, outputs=layer7)
    return model


# In[15]:


model = RNN()
model.summary()
# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# In[16]:


checkpoint = ModelCheckpoint('model-topcat-10-mix-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5',
                             verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
history = model.fit(x_train, y_train, batch_size=128, epochs=50,
                    validation_split=0.3, shuffle=True, callbacks=[checkpoint])


# In[17]:


# Save Complete Model
model.save('tc-nltk-lstm-rnn-10-mix.h5')

# # Save Model Configuration to JSON
# model_json = model.to_json()
# with open('tc-nltk-lstm-rnn-10-mix.json', 'w') as json_file:
#     json_file.write(model_json)

# # Load a Saved Model
# # model = load_model('tc-nltk-lstm-rnn.h5')


# In[22]:


# # Load Model Configuration from JSON
# json_file = open('tc-nltk-lstm-rnn-topcat-10-mix.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights('model-10-mix-009-0.917161-0.840018.h5') # Change before running
# loaded_model.save('tc-nltk-lstm-rnn-10-mix-weights.hdf5')
# loaded_model=load_model('tc-nltk-lstm-rnn-10-mix-weights.hdf5')

# Load Best Model
loaded_model = load_model('model-topcat-10-mix-020-0.764940-0.693318.h5')


# In[23]:


accr = model.evaluate(x_test, y_test, batch_size=256)
print()
print('Loss: {:0.3f}\tAccuracy: {:0.3f}'.format(accr[0], accr[1]))

accr2 = loaded_model.evaluate(x_test, y_test, batch_size=256)
print()
print('Loss: {:0.3f}\tAccuracy: {:0.3f}'.format(accr2[0], accr2[1]))


# In[ ]:


# plot_model(model, to_file='tc-nltk-lstm-4-model.png')


# In[24]:


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training and Validation Accuracy - LSTM RNN')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('tc-nltk-lstm-rnn-10-mix-acc.png', dpi=300, pad_inches=0.1)
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss - LSTM RNN')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('tc-nltk-lstm-rnn-10-mix-loss.png', dpi=300, pad_inches=0.1)
plt.show()


# In[ ]:
