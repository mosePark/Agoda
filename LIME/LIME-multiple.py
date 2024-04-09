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

# GOSDT
from sklearn.ensemble import GradientBoostingClassifier

from gosdt import GOSDT
from gosdt.model.threshold_guess import compute_thresholds


# language detect
from langdetect import detect
from langdetect import LangDetectException


def is_english_by_langdetect(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def predict_proba_reg(model, texts):
    """
    회귀 모델에 대한 predict_proba 함수의 대체.
    이 함수는 모델의 예측 값을 '확률'처럼 반환합니다.
    """
    preds = model.predict(texts)
    # 예측 값을 2차원 배열로 변환 (LIME이 예상하는 형태)
    return np.vstack([1-preds, preds]).T


def predict_proba_gpt(y_y_hat_series, texts):
    """
    GPT-3.5-turbo 모델의 예측값을 사용하는 predict_proba 함수입니다.
    """
    # 인덱스에 해당하는 예측 차이값(y-y_hat)을 찾습니다.
    preds = y_y_hat_series.loc[texts.index].values
    # preds 배열을 (n_samples, 2) 형태로 확장합니다.
    # 첫 번째 열은 1 - 예측값, 두 번째 열은 예측값입니다.
    # 이렇게 함으로써 각 샘플에 대한 확률이 되도록 합니다.
    return np.vstack([1 - preds, preds]).T







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


# # train, LIME
explainer = LimeTextExplainer(class_names=['Regression Output'])

# 큰 점수 차이가 나는 데이터들 확인 5점 초과

large_idx = df_eng[df_eng['y-y_hat'] > 5].index
common_idx = set(large_idx).intersection(X_test.index)

for idx in common_idx :
    exp = explainer.explain_instance(X_test[idx], lambda x: predict_proba_reg(pipeline, x), num_features=3)
    
    print(f'Explanation for index {idx}:')
    print(exp.as_list())

    exp.show_in_notebook(text=X_test[idx])

    print('\n' + '-'*80 + '\n')
