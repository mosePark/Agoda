'''
리뷰 텍스트만 넣고 점수 예측했을 때
예측이 매번 값이 조금 달라지는지 확인
'''

# %%
import os
import re

from matplotlib import font_manager, rc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# %%
# 한글 폰트 설정
font_path = 'C:/Windows/Fonts/NanumGothic.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# %%
# 문자열에서 숫자 추출 함수

def extract_number(text):
    match = re.search(r'\d+\.\d+', text)
    return float(match.group()) if match else None
# %%
os.chdir('C:/Users/mose/Desktop')

df1 = pd.read_csv("리뷰.csv")
df2 = pd.read_csv("24-07-10-리뷰 예측.csv")
df3 = pd.read_csv("24-07-12-리뷰 예측.csv")
df4 = pd.read_csv("24-07-14-리뷰 예측.csv")

# %%
# 빈 데이터프레임 생성
columns = ['hotel_name', 'Score', 'Country', 'Traveler Type', 'Room Type',
           'Stay Duration', 'Title', 'Text', 'Date']
df = pd.DataFrame(columns=columns)

# %%
# df4의 컬럼을 df로 복사
for column in columns:
    df[column] = df4[column]
# %%

# %%
df['y_hat'] = df1['y_hat'].apply(extract_number)
df['y_hat_'] = df2['y_hat'].apply(extract_number)
df['y_hat__'] = df3['y_hat'].apply(extract_number)
df['y_hat___'] = df4['y_hat'].apply(extract_number)

# %%
df.isnull().sum()
# %%
# 히스토그램 그리기
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
columns = ['y_hat', 'y_hat_', 'y_hat__', 'y_hat___']

for ax, column in zip(axes.flatten(), columns):
    sns.histplot(df[column], bins=range(1, 12), kde=True, ax=ax)
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.set_xticks(range(1, 11))

plt.tight_layout()
plt.show()

# %%

# 잔차 플롯을 위한 조합
pairs = [
    ('y_hat', 'y_hat_'),
    ('y_hat', 'y_hat__'),
    ('y_hat', 'y_hat___'),
    ('y_hat_', 'y_hat__'),
    ('y_hat_', 'y_hat___'),
    ('y_hat__', 'y_hat___')
]

# 잔차 플롯 생성
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for ax, (col1, col2) in zip(axes, pairs):
    residuals = df[col1] - df[col2]
    sns.scatterplot(x=df[col1], y=residuals, ax=ax)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title(f'Residual Plot: {col1} vs {col2}')
    ax.set_xlabel(col1)
    ax.set_ylabel('Residuals')

# 여백 설정을 조정하여 마이너스 기호가 잘리지 않도록 합니다.
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

# %%
