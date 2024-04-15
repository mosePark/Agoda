''' 아고다 리뷰에 대하여,
각 호텔별 리뷰 1개씩 가져와서 GPT를 활용해 리뷰 점수를 예측해본다.
"리뷰만 넣었을 때"와 "사용자정보, 호텔위치 등 정보를 추가한 리뷰 점수"
를 비교해본다.

약 446개 호텔의 리뷰를 예측하는 데 얼마나 걸리는지?
어떤 모델을 활용했는지?
토큰비용은 어떤지?

평가지표는 어떻게 활용할 것인지?
'''
import numpy as np
import pandas as pd

import os
import re

from openai import OpenAI
from langchain.chat_models import ChatOpenAI

open_api_key = 'your-api-key' # key_name = agoda-GPT

# ===== 데이터 로드
link = 'C:/Users/UOS/proj_0/preprocessing/'
df = pd.read_csv(link + 'agoda.csv', encoding = 'utf-8-sig')

# ===== 데이터 타입 확인 및 변경
df.dtypes

# ===== 리뷰 제목 뒤 " 기호 제거
df['Title'] = df['Title'].str[:-1]

# ===== 데이터 무작위 추출
len(df['hotel_name'].unique()) # 뉴욕 맨하튼 총 호텔 수
df['hotel_name'].unique()

random_reviews = df.groupby('hotel_name').agg(lambda x: np.random.choice(x)).reset_index()
# ===== 모델 아키텍쳐
'''
model : gpt-3.5-turbo
446개 리뷰 - $0.02
'''

def score_prediction_GPT(review_text) :
    chat_model = ChatOpenAI(openai_api_key = open_api_key, verbose = False)
    content = f"다음 리뷰를 읽고, 10점 만점으로 얼마나 높은 점수를 사용자가 줬을지 '숫자만으로' 대답해주세요.  리뷰: {review_text}"
    result = chat_model.predict(content)
    return result

score_prediction_GPT(random_reviews['Text'][0])

# ===== 모델링
y_hat = []
for i in range(446) :
    ans = score_prediction_GPT(random_reviews['Text'][0])
    y_hat.append(ans)

y = random_reviews['Score']

# combined_data = list(zip(y, y_hat))
# true_pred = pd.DataFrame(combined_data, columns=['Column1', 'Column2'])

# true_pred.to_excel("true_pred.xlsx", encoding = 'utf-8-sig')

# ===== 모델 평가

y_hat_ = []
for sen in y_hat :
    nums = re.findall(r'\b(?:[0-9]|10)(?:\.[0-9])?\b', sen)
    if nums :
        y_hat_.append(nums[-1])

len(y_hat_)

combined_data = list(zip(y, y_hat, y_hat_))
true_pred = pd.DataFrame(combined_data, columns=['true', 'y_hat', 'y_hat_'])

true_pred.to_excel("ture-pred-pred_.xlsx", encoding = 'utf-8-sig')

# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
# =====
