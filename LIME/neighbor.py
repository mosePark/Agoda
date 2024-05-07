'''
LIME
'''

import numpy as np
import pandas as pd

import os



os.chdir('C:/Users/UOS/Desktop/Agoda-Data/raw')

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

explainer = LimeTextExplainer(class_names = cls_nms)
exp = explainer.explain_instance(X_test[5407], pipeline.predict_proba)

exp.show_in_notebook(text=True)

from tqdm import tqdm
import time

differ_idx = []
for idx in tqdm(X_test.index, desc="Processing"):
    exp = explainer.explain_instance(X_test[idx], pipeline.predict_proba)
    what = cls_nms[pipeline.predict([X_test[idx]]).reshape(1,-1)[0,0]]
    
    if what == "difference":
        differ_idx.append(idx)
        exp.show_in_notebook(text=True)
        

