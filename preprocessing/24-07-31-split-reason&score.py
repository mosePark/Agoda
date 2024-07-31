'''
LLM 대답에 이유가 추가된 예측자료를 점수와 이유 변수로 나눈 전처리 코드
'''

#%%
import os
import re
import numpy as np
import pandas as pd

#%%
# 사용자함수
def extract_score_reason(text):
    score = None
    reason = None
    
    # 점수 추출
    score_match = re.search(r'(\d+(\.\d+)?(/[1-9]\d?)?)', text)
    if score_match:
        score = score_match.group(1)
        if '/' in score:
            score = score.split('/')[0]  # 2/10과 같은 형식을 처리
        score = int(float(score))  # 실수 형식을 처리하고 정수로 변환

    # 이유 추출
    if 'Reason:' in text:
        reason = text.split('Reason:')[1].strip()
    elif '\n' in text:
        reason = text.split('\n', 1)[1].strip()
    elif '. ' in text:
        reason = text.split('. ', 1)[1].strip()
    
    return score, reason

# %%
'''
데이터 로드
'''

os.chdir('C:/Users/mose/agoda/data/')

df = pd.read_csv("Review+Reason_Final.csv")

#%%
# 새로운 칼럼 추가
df['pred'], df['Reason'] = zip(*df['y_hat'].apply(extract_score_reason))

#%%
df.isnull().sum()

#%%
df[df['pred'].isnull() | df['Reason'].isnull()]

#%%
df[df['Text'].isnull()]
'''
데이터 조회해보니까, Text가 결측인건 그냥 지워도 되겠음. 예측도 잘 못함
정보가 없으니 8~9점 짜리도 1점으로 예측
앞으로 Text가 없는 데이터는 삭제해서 접근.
'''

#%%
len(df) - len(df.dropna(subset=['Text'])) # 27이 나와야 딱 맞음

df = df.dropna(subset=['Text'])

#%%
# 결측 재조회
df.isnull().sum()

#%%
# pred, reason 결측인 데이터 조회해보기

# df[df['pred'].isnull() | df['Reason'].isnull()]

'''
조회해보니 제목과 텍스트를 prompt에 넣어서 예측해야하는 것으로 판단.
'''

#%%

df = df.dropna(subset=['Reason'])
df = df.dropna(subset=['pred'])

# 결측 재조회
df.isnull().sum()

df.to_csv("reason.csv", index=False, encoding='utf-8-sig')
