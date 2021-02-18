
# coding: utf-8

# In[1]:


# Text Classification / tc-reuter.ipynb
# Gourav Siddhad
# 07-Mar-2019


# In[34]:


print('Importing Libraries', end='')

import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle

from bs4 import BeautifulSoup
import string
import csv
import os
import itertools
import tarfile
import re
import numpy as np
import time

print(' - Done')


# # Read Data

# In[44]:


if not os.path.exists('reuters-data'):
    os.mkdir('reuters-data')

# Decompress data
# compressed = tarfile.open('reuters-data/reuters21578.tar.gz')
# compressed.extractall(path = 'reuters-data/')
# compressed.close()

# List the uncompressed data
print('Directory Listing')
os.listdir('reuters-data/')


# In[45]:


# Load in each data file
text_data = []

print('Loading Files - ', end=' ')
for index in range(22):
    if index is not 17:
        print(index, end=' ')
        filename = 'reuters-data/reut2-{0}.sgm'.format(str(index).zfill(3))
        with open(filename, 'r', encoding = 'utf-8', errors = 'ignore') as infile:
            text_data.append(infile.read())

print()
print('Loading Files - Done')


# In[69]:


import pandas as pd

# Separate each text file into articles

print('Converting Files to Articles - ', end='')
i, counter = 0, 0

columns = ['Title', 'Topics', 'Text']
index = [x for x in range(0, 18103)]
df = pd.DataFrame(columns=columns, index=index)

for textfile in text_data:
    print(i, end=' ')
    # Parse text as html
    soup = BeautifulSoup(textfile, 'html.parser')
    
    # Extract article between <BODY> and </BODY> and convert to standard text. Add to list of articles
    title = soup.find('reuters')['lewissplit']
    topics = soup.find_all('topics')
    
    j=0
    for article in soup.find_all('body'):
        df['Text'][counter] = article.get_text()
        df['Title'][counter] = title
        df['Topics'][counter] = topics[j].find_all('d')
        j += 1
        counter += 1
    
    i += 1

print()
print('Files to Aritcle Conversion - Done')
print('Total Articles - ', counter)


# In[79]:


df


# In[78]:


count = 0
for topic in df['Topics']:
    if len(topic)>0:
        count += 1
print('Documents with Complete Details - ', count)

