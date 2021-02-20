# coding: utf-8

# Text Classification / tc-nltk-lsa-rf.ipynb
# Gourav Siddhad
# 05-Mar-2019

print('Importing Libraries', end='')

from datetime import datetime
import time
import pandas as pd
import numpy as np
import re
from itertools import cycle
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import reuters
import nltk

print(' - Done')


# # Preprocessing


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


def feature_values(doc, representer):
    doc_representation = representer.transform([doc])
    features = representer.get_feature_names()
    return [(features[index], doc_representation[0, index]) for index in doc_representation.nonzero()[1]]


print('Sorting Train:Test Docs', end='')
X_train, X_test, y_train, y_test = train_test_split(
    all_docs, all_labels, test_size=0.2, random_state=42)
print(' - Done')

print('Creating TF-IDF (Train)', end='')
tfidf = TfidfVectorizer(tokenizer=tokenize, use_idf=True,
                        sublinear_tf=True, norm='l2')
train_features = tfidf.fit_transform(X_train)
print(' - Done')

print('Creating TF-IDF (Test)', end='')
test_features = tfidf.transform(X_test)
print(' - Done')

print('Binarizing MultiLabels', end='')
lb = MultiLabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)
print(' - Done')

del all_docs
del all_labels


# ## Latent Semantic Analysis


def lsa_svd(comp=5000):
    svd = TruncatedSVD(n_components=comp)
    lsa = make_pipeline(svd, Normalizer(copy=False))

    lsa.fit(train_features)
    train_lsa = lsa.transform(train_features)
    test_lsa = lsa.transform(test_features)

    return (sum(svd.explained_variance_ratio_), train_lsa, test_lsa)


# # Training and Testing


print('Training & Testing Random Forest')
print('Iterations - ', end='')

variance_rf, acc_rf, train_time_rf, test_time_rf = [], [], [], []
index = []
maxn, step = 200, 1

i = step
while i <= maxn:
    print(i, end='')
    index.append(i)
    (variance, train_lsa, test_lsa) = lsa_svd(i)

    model_rf = RandomForestClassifier(n_estimators=i)
    start = time.time()
    model_rf.fit(train_lsa, y_train)
    end = time.time()

    train_time_rf.append(end-start)
    print('.', end=' ')

    start = time.time()
    y_pred = model_rf.predict(test_lsa)
    end = time.time()

    test_time_rf.append(end-start)
    acc_rf.append(accuracy_score(y_test, y_pred))
    variance_rf.append(variance)

    i += step

    del model_rf
    del start
    del end
    del y_pred
    del variance
    del train_lsa
    del test_lsa


# # Results


print('Creating Result DataFrame', end=' ')
columns = ['Variance', 'Accuracy', 'Train Time', 'Test Time']
df_rf = pd.DataFrame(columns=columns, index=index)

df_rf['Variance'] = variance_rf
df_rf['Accuracy'] = acc_rf
df_rf['Train Time'] = train_time_rf
df_rf['Test Time'] = test_time_rf

print(' - Done')

print()
print(df_rf)


print('Extracted Data Shape ')
print('Train : ', train_features.shape)
print('Test  : ', test_features.shape)
print()
plt.plot(index, acc_rf, label='Accuracy')
plt.legend(loc=4)
plt.title('Accuracy')
plt.savefig('tc-nltk-lsa-rf.png', dpi=150, pad_inches=0.1)
plt.show()


print(datetime.now())

print('Writing to CSV', end='')
df_rf.to_csv('tc-nltk-lsa-rf.csv', header=True)
print(' - Done')


maxa, mindex = 0, 0
i = step
while i <= maxn:
    if df_rf['Accuracy'][i] > maxa:
        maxa = df_rf['Accuracy'][i]
        mindex = i
    i += step

print('Max - ', maxa)
print('Index - ', mindex)


# ## Training Testing and Results on Best Accuracy


print('Training Random Forest', end='')

# Max -  0.726598702502317
# Index -  37
# mindex = 37

(variance, train_lsa, test_lsa) = lsa_svd(mindex)
model_rf = RandomForestClassifier(n_estimators=i)  # Use optimum values

start = time.time()
model_rf.fit(train_lsa, y_train)
end = time.time()
print(' - Done')

train_time_rf = end-start

print('Testing Random Forest', end='')
start = time.time()
y_pred = model_rf.predict(test_lsa)
end = time.time()
print(' - Done')

test_time_rf = end-start
acc_rf = accuracy_score(y_test, y_pred)
cmat_rf = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))


def plot_roc(title, test_y, pred_y):
    print('ROC for : ', title)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_class = test_y.shape[1]
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], pred_y[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), pred_y.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

    # Interpolate all ROC curves
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Average it and compute AUC
    mean_tpr /= n_class

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    ax1.plot(fpr["micro"], tpr["micro"],
             label='micro-avg ROC (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink')
    ax1.plot(fpr["macro"], tpr["macro"],
             label='macro-avg ROC (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC multi-class')
    ax1.legend(loc="lower right")

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_class), colors):
        ax2.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    ax2.set_title('ROC Individual Classes')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC multi-class')
    plt.savefig(title+'.png', dpi=300, pad_inches=0.1)
    plt.show()

    return (roc_auc)


roc_auc_rf = plot_roc('tc-nltk-lsa-rf-accurate', y_test, y_pred)


print('Original Data Shape ')
print('Train : ', train_features.shape)
print('Test  : ', test_features.shape)
print('Reduced Data Shape ')
print('Train : ', train_lsa.shape)
print('Test  : ', test_lsa.shape)
print()
print('Accuracy :', acc_rf * 100)
print(
    'Time Taken - Train : {0:.4f}, Test : {1:.4f}'.format(train_time_rf, test_time_rf))
print('Area under ROC : {0:7.4f}'.format(roc_auc_rf['micro']*100))
print()
print('Confusion Matrix')
print(cmat_rf)
