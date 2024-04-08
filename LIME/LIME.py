# stats
import numpy as np
import pandas as pd

# WD
import os

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# LIME
from lime.lime_text import LimeTextExplainer


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

# data split
X_train, X_test, y_train, y_test = train_test_split(df_eng['Text'], df_eng['y-y_hat'], random_state=240408)

# 
pipeline = make_pipeline(TfidfVectorizer(), LinearRegression())
pipeline.fit(X_train, y_train)


# train, LIME
explainer = LimeTextExplainer(class_names=['Regression Output'])
idx = 0  # 설명하고 싶은 테스트 데이터의 인덱스
exp = explainer.explain_instance(X_test.iloc[idx], pipeline.predict, num_features=6)

