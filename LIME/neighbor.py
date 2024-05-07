'''
LIME
'''

import numpy as np
import pandas as pd

import os



os.chdir('.../data/')

eng = pd.read_csv("eng.csv", index_col=0)


# Count vec
from sklearn.feature_extraction.text import CountVectorizer

cntvectorizer = CountVectorizer(
    stop_words='english'
)


# tf-idf vec
from sklearn.feature_extraction.text import TfidfVectorizer

tfidfvectorizer = TfidfVectorizer(
    stop_words='english'
)


# hashing vec
from sklearn.feature_extraction.text import HashingVectorizer

hashingvectorizer = HashingVectorizer()




'''
neighborhood analysis (LIME)

neighbor point sampling
'''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


X_train, X_test, y_train, y_test = train_test_split(eng['Text'], eng['difference'], random_state=240507)

# randomforest
pipeline = make_pipeline(cntvectorizer, RandomForestClassifier())
pipeline.fit(X_train, y_train)

# evaluation
print('test accuracy', accuracy_score(y_test, pipeline.predict(X_test)))
print('Confusion Matrix:\n\n', confusion_matrix(y_test, pipeline.predict(X_test)))


# train, LIME
from lime.lime_text import LimeTextExplainer

cls_nms = ['Not', 'difference']


from tqdm import tqdm
import time

differ_idx = []
for idx in tqdm(X_test.index, desc="Processing"):
    exp = explainer.explain_instance(X_test[idx], pipeline.predict_proba)
    what = cls_nms[pipeline.predict([X_test[idx]]).reshape(1,-1)[0,0]]
    
    if what == "difference":
        differ_idx.append(idx)
        exp.show_in_notebook(text=True)


differ_text = []
for idx in differ_idx :
    differ_text.append(X_test[idx])
    print(idx, ' : ',X_test[idx], ' : ' , eng['difference'][idx])

differ = [11951, 2839, 21212, 10495, 20344, 10250]

# EDA true data
train = pd.read_csv("train.csv", index_col=0)
differ_train = train.loc[differ]

import openpyxl
differ_train.to_excel("differ_train.xlsx")

eng['Text'][20344]
