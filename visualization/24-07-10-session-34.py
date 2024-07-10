'''
session-34
리뷰
국가 리뷰
여행객 리뷰
국가 여행객 리뷰

각 예측 점수를 비교해보는 작업

일단 이 파일은 전처리

+ 시각화도 추가함
'''
import re
import os

import itertools

import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates

import matplotlib.pyplot as plt
import seaborn as sns



def extract_number(text):
    match = re.search(r'\d+\.\d+', text)
    return float(match.group()) if match else None

os.chdir('C:/Users/mose/Desktop/')

df1 = pd.read_csv("리뷰.csv")
df2 = pd.read_csv("국가+리뷰.csv")
df3 = pd.read_csv("여행객+리뷰.csv")
df4 = pd.read_csv("국가+리뷰+여행객.csv")

df1.columns
df2.columns
df3.columns
df4.columns

# 빈 데이터프레임 생성
columns = ['hotel_name', 'Score', 'Country', 'Traveler Type', 'Room Type',
           'Stay Duration', 'Title', 'Text', 'Date']
df = pd.DataFrame(columns=columns)

# df4의 컬럼을 df로 복사
for column in columns:
    df[column] = df4[column]

df['y_hat'] = df1['y_hat'].apply(extract_number)
df['y_hat_'] = df2['y_hat__'].apply(extract_number)
df['y_hat__'] = df3['y_hat__'].apply(extract_number)
df['y_hat___'] = df4['y_hat__'].apply(extract_number)

df.isnull().sum()

score = df.iloc[:, -4:]

# 박스 플롯
plt.figure(figsize=(10, 6))
sns.boxplot(data=score)
plt.title('Box Plot of Predicted Scores')
plt.show()

# 밀도 플롯
plt.figure(figsize=(10, 6))
sns.kdeplot(score['y_hat'], label='y_hat', shade=True)
sns.kdeplot(score['y_hat_'], label='y_hat_', shade=True)
sns.kdeplot(score['y_hat__'], label='y_hat__', shade=True)
sns.kdeplot(score['y_hat___'], label='y_hat___', shade=True)
plt.title('Density Plot of Predicted Scores')
plt.legend()
plt.show()

# Bland-Altman plot 함수 정의
def bland_altman_plot(data1, data2, ax=None):
    mean = (data1 + data2) / 2
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    loa_upper = md + 1.96 * sd
    loa_lower = md - 1.96 * sd

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(mean, diff, alpha=0.5)
    ax.axhline(md, color='gray', linestyle='--')
    ax.axhline(loa_upper, color='red', linestyle='--')
    ax.axhline(loa_lower, color='red', linestyle='--')
    ax.set_xlabel('Mean of Two Measurements')
    ax.set_ylabel('Difference between Measurements')
    ax.set_title('Bland-Altman Plot')

    # 동의 한계 내외 데이터 포인트 인덱스 계산
    outside_loa_indices = ((diff < loa_lower) | (diff > loa_upper))
    
    return df.index[outside_loa_indices]

# %%
# 예측값 쌍 목록 생성
predictions = ['y_hat', 'y_hat_', 'y_hat__', 'y_hat___']
pairs = list(itertools.combinations(predictions, 2))

# Bland-Altman 플롯 그리기 및 바깥 데이터 포인트 인덱스 저장
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

outside_indices_dict = {}

for (pred1, pred2), ax in zip(pairs, axes):
    outside_loa_indices = bland_altman_plot(df[pred1], df[pred2], ax=ax)
    outside_indices_dict[(pred1, pred2)] = set(outside_loa_indices)
    ax.set_title(f'Bland-Altman Plot: {pred1} vs {pred2}')

plt.tight_layout()
plt.show()


# %%
# Case 1 : 리뷰 vs 국가+리뷰

# y_hat과 y_hat_ 쌍에 대한 동의 한계 바깥에 있는 인덱스 추출
indices_y_hat_y_hat_ = outside_indices_dict[('y_hat', 'y_hat_')]

# 교집합 인덱스를 이용해 데이터프레임에서 해당 데이터 포인트 추출
outside_loa_data_y_hat_y_hat_ = df.loc[list(indices_y_hat_y_hat_)]


# Case 2 : 리뷰 vs 여행객+리뷰

# y_hat과 y_hat__ 쌍에 대한 동의 한계 바깥에 있는 인덱스 추출
indices_y_hat_y_hat__ = outside_indices_dict[('y_hat', 'y_hat__')]

# 교집합 인덱스를 이용해 데이터프레임에서 해당 데이터 포인트 추출
outside_loa_data_y_hat_y_hat__ = df.loc[list(indices_y_hat_y_hat__)]

# Case 3 : 리뷰 vs 국가+여행객+리뷰

# y_hat과 y_hat___ 쌍에 대한 동의 한계 바깥에 있는 인덱스 추출
indices_y_hat_y_hat___ = outside_indices_dict[('y_hat', 'y_hat___')]

# 교집합 인덱스를 이용해 데이터프레임에서 해당 데이터 포인트 추출
outside_loa_data_y_hat_y_hat___ = df.loc[list(indices_y_hat_y_hat___)]


# 교집합 인덱스를 이용해 데이터프레임에서 해당 데이터 포인트 추출
common_indices = indices_y_hat_y_hat_.intersection(indices_y_hat_y_hat__).intersection(indices_y_hat_y_hat___)
outside_loa_common_data = df.loc[list(common_indices)]

outside_loa_common_data.to_csv("3가지케이스 교집합 인덱스.csv", encoding='utf-8-sig')

# %%
## outside_loa_common_data 분석해보기

# 'Score' 열에 대한 박스플롯 그리기
outside_loa_common_data['Score'].plot(kind='box')
plt.title('Boxplot of Score in Common Indices Outside LOA')
plt.ylabel('Score')
plt.show()



# %%
# 전체 vs 리뷰 교집합 국가+리뷰 차이 보기

# 'Country' 테이블
country_table = outside_loa_common_data['Country'].value_counts()
print(country_table.head(10))

country_df = df['Country'].value_counts()
print(country_df.head(10))

# 'Country' 열에 대한 비율 테이블 작성
country_table = outside_loa_common_data['Country'].value_counts(normalize=True)
print(country_table.head(10))

# 전체 데이터프레임에서 'Country' 열에 대한 비율 테이블 작성
country_df = df['Country'].value_counts(normalize=True)
print(country_df.head(10))

# %%

# %%
# 전체 vs 리뷰 교집합 여행객+리뷰  테이블 차이 보기

# 'Traveler Type' 테이블
traveler_type_table = outside_loa_common_data['Traveler Type'].value_counts()
print(traveler_type_table)

traveler_type_df = df['Traveler Type'].value_counts()
print(traveler_type_df)

# 'Traveler Type' 열에 대한 비율 테이블 작성
traveler_type_ratio_table = outside_loa_common_data['Traveler Type'].value_counts(normalize=True)
print(traveler_type_ratio_table)

# 전체 데이터프레임에서 'Traveler Type' 열에 대한 비율 테이블 작성
traveler_type_df_ratio = df['Traveler Type'].value_counts(normalize=True)
print(traveler_type_df_ratio)

# %%
os.chdir("C:/Users/mose/agoda/data/")

df = pd.read_csv("train.csv")

df.dtypes

df['Score']


# 히스토그램
plt.figure(figsize=(12, 6))
sns.histplot(df['Score'], bins=10, kde=True)
plt.title('Score Distribution - Histogram')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()
