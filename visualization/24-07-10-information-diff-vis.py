'''
리뷰 예측점수
리뷰+정보 예측점수 등 총 3가지 비교해서 시각화한 것

결론 : 차이가 없다.
'''



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import os

from google.colab import drive
drive.mount('/content/drive')

def extract_number(text):
    match = re.search(r'\d+\.\d+', text)
    return float(match.group()) if match else None

os.chdir('/content/drive/MyDrive/agoda/')

df2 = pd.read_csv("국가+리뷰.csv")
df3 = pd.read_csv("여행객+리뷰.csv")
df4 = pd.read_csv("국가+리뷰+여행객.csv")

# 빈 데이터프레임 생성
columns = ['hotel_name', 'Score', 'Country', 'Traveler Type', 'Room Type',
           'Stay Duration', 'Title', 'Text', 'Date']
df = pd.DataFrame(columns=columns)

# df4의 컬럼을 df로 복사
for column in columns:
    df[column] = df4[column]

df['y_hat_'] = df2['y_hat__'].apply(extract_number)
df['y_hat__'] = df3['y_hat__'].apply(extract_number)
df['y_hat___'] = df4['y_hat__'].apply(extract_number)

df.head()

score = df.iloc[:, -3:]

# 박스 플롯
plt.figure(figsize=(10, 6))
sns.boxplot(data=score)
plt.title('Box Plot of Predicted Scores')
plt.show()

# 밀도 플롯
plt.figure(figsize=(10, 6))
sns.kdeplot(score['y_hat_'], label='y_hat_', shade=True)
sns.kdeplot(score['y_hat__'], label='y_hat__', shade=True)
sns.kdeplot(score['y_hat___'], label='y_hat___', shade=True)
plt.title('Density Plot of Predicted Scores')
plt.legend()
plt.show()
