%load_ext autoreload
%autoreload 2
import os
import os.path
import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble
import spacy
import sys
from sklearn.feature_extraction.text import CountVectorizer
from anchor import anchor_text
import time

# language detect
from langdetect import detect
from langdetect import LangDetectException

def is_english_by_langdetect(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False


os.chdir('C:/Users/UOS/Desktop/Agoda-Data/raw')

train = pd.read_csv("train.csv", index_col=0)

# label setting
train['y-y_hat'] = np.abs(train['Score'] - train['y_hat_'])

# exctract data
df = train.loc[:, ['Text', 'y-y_hat']]

# only English
df_eng = df[df['Text'].apply(is_english_by_langdetect)]

# y : binary
df_eng['y-y_hat'] = np.abs(df_eng['y-y_hat']).apply(lambda x: 1 if x >= 3 else 0)



'''
Anchor
'''


data = df_eng['Text'].apply(lambda x: x.encode('utf-8')).tolist()
labels = df_eng['y-y_hat'].tolist()

train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(data, labels, test_size=.2, random_state=42)
train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1, random_state=42)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)

vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)
train_vectors = vectorizer.transform(train)
test_vectors = vectorizer.transform(test)
val_vectors = vectorizer.transform(val)

c = sklearn.ensemble.RandomForestClassifier(n_estimators=500, n_jobs=10)
c.fit(train_vectors, train_labels)
preds = c.predict(val_vectors)
print('Val accuracy', sklearn.metrics.accuracy_score(val_labels, preds))
def predict_lr(texts):
    return c.predict(vectorizer.transform(texts))

'''
Explaning a prediction
'''
nlp = spacy.load('en_core_web_sm')

explainer = anchor_text.AnchorText(nlp, ['not', 'distinct'], use_unk_distribution=True)

np.random.seed(1)
text = 'Nice views.'
pred = explainer.class_names[predict_lr([text])[0]]
alternative =  explainer.class_names[1 - predict_lr([text])[0]]
print('Prediction: %s' % pred)
exp = explainer.explain_instance(text, predict_lr, threshold=0.95)

print('Anchor: %s' % (' AND '.join(exp.names())))
print('Precision: %.2f' % exp.precision())
print()
print('Examples where anchor applies and model predicts %s:' % pred)
print()
print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
print()
print('Examples where anchor applies and model predicts %s:' % alternative)
print()
print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_different_prediction=True)]))
