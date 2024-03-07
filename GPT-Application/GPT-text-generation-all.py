''' 
모든 데이터의 리뷰를 GPT에 넣어 리뷰 점수 예측하기
'''
import numpy as np
import pandas as pd

import os
import re

'''
데이터 로드
'''

os.chdir('/.../Agoda-Data/raw')

df = pd.read_csv("agoda.csv", index_col=0)
df.head()

df.dtypes


===== 아래 코드는 GPT를 활용해 prediction 예측하는 코드

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI



'''
api key 로드
'''
load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
print(API_KEY)

# 리뷰 제목 뒤 " 기호 제거
df['Title'] = df['Title'].str[:-1]

# ===== 모델 아키텍쳐
'''
model : gpt-3.5-turbo
총 리뷰 수 : 32084-27=23057
'''

def score_prediction_GPT(review_text) :
    chat_model = ChatOpenAI(openai_api_key = API_KEY, verbose = False)
    content = f"어떤 유저가 작성한 리뷰 텍스트에 대해 10점 만점에 몇점을 줬을지 예측해줘. 단, 소수점 첫째자리까지이고 답변 예시는 '점수 : 9.4' \n 리뷰 텍스트: {review_text}"
    result = chat_model.predict(content)
    return result


y_hat = []
for i in range(len(df)) :
    ans = score_prediction_GPT(df['Text'][i])
    y_hat.append(ans)
    pd.DataFrame(y_hat).to_excel("y_hat.xlsx")

df['y_hat'] = y_hat

df.to_csv("agoda+y_hat.csv", encoding='utf-8-sig')

def extract_score(text):
    # 정규식을 사용하여 부동 소수점 형식의 숫자 패턴을 찾음
    pattern = r'\d+(\.\d+)?'  # 소수점을 포함할 수 있는 패턴
    match = re.search(pattern, text)
    
    if match:
        # 숫자가 발견되면 해당 숫자를 반환
        return float(match.group())
    else:
        # 숫자가 없을 경우 None 반환
        return None
    
y_hat_value = []

for i in range(len(df)) :
    y_hat_value.append(extract_score(df['y_hat'][i]))

df['score_pred'] = y_hat_value
