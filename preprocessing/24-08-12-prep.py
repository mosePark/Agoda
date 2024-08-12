'''
대대적인 전처리 작업 시작

작업 목록
1. title, text " 제거
2. title, text 합치는 text로
3. 결측치, 중복행 제거
'''
#%%
def remove_trailing_quote(text):
    if isinstance(text, str):
        return text.rstrip('”')
    return text


#%%
import re
import os
import numpy as np
import pandas as pd

os.chdir('C:/Users/UOS/proj_0/preprocessing/')

df = pd.read_csv("agoda.csv")

df.rename(columns={'hotel_name': 'Hotel'}, inplace=True)

df.isnull().sum()


#%% 작업 1

df['Title'] = df['Title'].apply(remove_trailing_quote)
df['Text'] = df['Text'].apply(remove_trailing_quote)

df.head()

#%% 작업 2

df['Full'] = df['Title'].fillna('') + "\n" + df['Text'].fillna('')
df['Full'] = df['Full'].replace("\n", np.nan)

df = df.dropna(subset=['Full']) # idx = 1945 제거, FUll = NaN
#%% 작업 3
 
# 중복 데이터 (행) 조회 및 제거
df_dup = df[df.duplicated(subset=['Hotel', 'Score','Country', 'Traveler Type', 'Stay Duration', 'Title', 'Text'], keep=False)]
df_dup

# 중복된 행 중 첫 번째 행의 인덱스 저장
first_indices = df_dup.index

# 중복된 행 제거하고 각 그룹의 첫 번째 행만 남기기
df_unique = df.drop_duplicates(subset=['Hotel', 'Score', 'Country', 'Traveler Type', 'Stay Duration', 'Title', 'Text'], keep='first')


# %%
