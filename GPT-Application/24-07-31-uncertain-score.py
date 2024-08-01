#%%

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

#%%

# 문자열 데이터를 리스트로 변환하고 각 원소를 float으로 변환하는 함수 정의
def convert_to_float_list(string_data):
    try:
        # 문자열을 리스트로 변환
        data_list = json.loads(string_data)
        # 리스트의 각 원소를 float으로 변환
        float_list = [float(i) for i in data_list]
        return float_list
    except json.JSONDecodeError:
        return []
    
# 데이터를 청크로 나누어 코사인 유사도를 계산하는 함수 정의
def calculate_cosine_similarity_in_chunks(embd1, embd2, chunk_size=1000):
    cosine_similarities = []
    for start in range(0, len(embd1), chunk_size):
        end = min(start + chunk_size, len(embd1))
        chunk_cosine_similarities = cosine_similarity(embd1[start:end], embd2[start:end])
        cosine_similarities.extend(chunk_cosine_similarities.diagonal())
    return cosine_similarities

# 부트스트랩을 통해 불확실성 계산 함수
def bootstrap_threshold(data, threshold, n_bootstrap=1000):
    n = len(data)
    bootstrap_proportions = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        proportion = np.sum(sample > threshold) / n
        bootstrap_proportions.append(proportion)
    return np.array(bootstrap_proportions)
#%%

os.chdir('C:/Users/mose/agoda/data/')

df1 = pd.read_csv("24-07-29-임베딩추가.csv", encoding='utf-8-sig')
df2 = pd.read_csv("24-07-31-r-임베딩추가.csv", encoding='utf-8-sig')

#%%

len(df1)
len(df2)
#%%

df1.isnull().sum()
df2.isnull().sum()
#%%

# 특정칼럼명 변경
df1.rename(columns={'embedding': 'embd'}, inplace=True)

#%%

df1.columns
df2.columns

#%%
df1 = df1.dropna(subset=['Text'])

#%%
len(df1), len(df2)

#%%
# 데이터프레임의 각 행에 대해 변환 함수 적용
df1['embd'] = df1['embd'].apply(convert_to_float_list)
df2['r-embd'] = df2['r-embd'].apply(convert_to_float_list)

# %%
embd1 = df1['embd'].to_list()
embd2 = df2['r-embd'].to_list()

# 코사인 유사도 계산
cosine_similarities = calculate_cosine_similarity_in_chunks(embd1, embd2, chunk_size=1000)

len(cosine_similarities)

df2['cosine_similarity'] = cosine_similarities

# %%
# %%
# 코사인 유사도의 빈도를 0.1 단위로 나누어 시각화
bins = np.arange(0, 1.1, 0.1)  # 0.1 간격으로 구간 생성

# 히스토그램 그리기
counts, bins, patches = plt.hist(cosine_similarities, bins=bins, edgecolor='black', alpha=0.7)

# 각 막대 위에 비율 표시
for count, bin, patch in zip(counts, bins, patches):
    height = patch.get_height()
    plt.text(patch.get_x() + patch.get_width() / 2, height, f'{count / len(cosine_similarities):.2%}', 
             ha='center', va='bottom')

plt.title('Histogram of Cosine Similarities')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.xticks(bins)  # x축 눈금을 구간에 맞춰 설정
plt.grid(True)
plt.show()
# %%
# 임계값 설정
threshold = 0.5

# 부트스트랩을 통해 임계값을 넘는 비율과 신뢰 구간 계산
bootstrap_samples = bootstrap_threshold(cosine_similarities, threshold)
mean_proportion = np.mean(bootstrap_samples)
conf_interval = np.percentile(bootstrap_samples, [2.5, 97.5])

print(f'Mean Proportion above threshold {threshold}: {mean_proportion}')
print(f'95% Confidence Interval: {conf_interval}')
# %%

# 히스토그램 그리기
bins = np.arange(0, 1.1, 0.1)
counts, bins, patches = plt.hist(cosine_similarities, bins=bins, edgecolor='black', alpha=0.7)

# 각 막대 위에 비율 표시
total = len(cosine_similarities)
for count, bin, patch in zip(counts, bins[:-1], patches):
    height = patch.get_height()
    plt.text(patch.get_x() + patch.get_width() / 2, height, f'{count / total:.2%}', 
             ha='center', va='bottom')

plt.title('Histogram of Cosine Similarities with Uncertainty')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.xticks(bins)  # x축 눈금을 구간에 맞춰 설정
plt.grid(True)

# 신뢰 구간 시각화
plt.axvline(threshold, color='r', linestyle='dashed', linewidth=1, label=f'Threshold {threshold}')
plt.axvline(conf_interval[0], color='g', linestyle='dashed', linewidth=1, label='95% CI Lower')
plt.axvline(conf_interval[1], color='g', linestyle='dashed', linewidth=1, label='95% CI Upper')

plt.legend()
plt.show()
# %%
# 유사도가 낮은 필터링된 데이터의 MSE는?
# 조건을 만족하는 데이터 필터링
filtered_df = df2[df2['cosine_similarity'] < 0.2]

# MSE 계산
mse = mean_squared_error(filtered_df['Score'], filtered_df['pred'])

print(f'Mean Squared Error (MSE) for cosine_similarity < 0.2: {mse}')
# %%

# 조건을 만족하는 데이터 필터링
filtered_df_above = df2[df2['cosine_similarity'] >= 0.2]

# 'Score'와 'pred' 간의 차이 계산
differences_above = filtered_df_above['Score'] - filtered_df_above['pred']

# MSE 계산
mse_above = mean_squared_error(filtered_df_above['Score'], filtered_df_above['pred'])

print(f'Mean Squared Error (MSE) for cosine_similarity >= 0.2: {mse_above}')
# %%

# 구간 설정
bins = np.arange(0, 1.1, 0.1)  # [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]

# 구간별 MSE 계산
mse_by_bin = {}

for i in range(len(bins) - 1):
    lower_bound = bins[i]
    upper_bound = bins[i + 1]
    # 구간 내 데이터 필터링
    bin_data = df2[(df2['cosine_similarity'] >= lower_bound) & (df2['cosine_similarity'] < upper_bound)]
    
    if not bin_data.empty:
        # MSE 계산
        mse = mean_squared_error(bin_data['Score'], bin_data['pred'])
        mse_by_bin[f'{lower_bound}-{upper_bound}'] = mse

# 결과 출력
for bin_range, mse in mse_by_bin.items():
    print(f'MSE for cosine_similarity in range {bin_range}: {mse}')

#%% 추가파트 cosine vs MSE 꺾은선
# 주어진 데이터
similarity_ranges = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9']
mse_values = [8.59, 6.06, 4.48, 3.35, 2.90, 3.07, 3.33, 3.43, 1.57]

# x축 레이블을 숫자형으로 변환
x = [i for i in range(len(similarity_ranges))]

# 꺾은선 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(x, mse_values, marker='o', linestyle='-', color='b')

# x축 레이블 설정
plt.xticks(x, similarity_ranges, rotation=45)
plt.xlabel('Cosine Similarity Range')
plt.ylabel('MSE')
plt.title('MSE by Cosine Similarity Range')
plt.grid(True)

# 그래프 표시
plt.tight_layout()
plt.show()
# %% 

# 코사인 유사도 구간 설정
bins = np.arange(0, 1.1, 0.1)
labels = [f'{round(b, 1)}-{round(b+0.1, 1)}' for b in bins[:-1]]

# 코사인 유사도 구간에 따라 범주형 변수 생성
df2['cosine_bin'] = pd.cut(df2['cosine_similarity'], bins=bins, labels=labels, include_lowest=True)

# 박스 플롯 그리기
plt.figure(figsize=(12, 8))
boxplot = df2.boxplot(column='Score', by='cosine_bin', grid=False, showmeans=True)

# 플롯 제목 및 축 레이블 설정
plt.title('Distribution of Scores by Cosine Similarity Range')
plt.suptitle('')  # 기본 제목 제거
plt.xlabel('Cosine Similarity Range')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.grid(True)

# 그래프 표시
plt.tight_layout()
plt.show()
# %%
# 실제 점수 구간 설정 (1점부터 10점까지)
bins = np.arange(1, 12, 1)
labels = [f'{int(b)}' for b in bins[:-1]]

# 실제 점수 구간에 따라 범주형 변수 생성
df2['score_bin'] = pd.cut(df2['Score'], bins=bins, labels=labels, include_lowest=True, right=False)

# 박스 플롯 그리기
plt.figure(figsize=(12, 8))
boxplot = df2.boxplot(column='cosine_similarity', by='score_bin', grid=False, showmeans=True)

# 플롯 제목 및 축 레이블 설정
plt.title('Distribution of Cosine Similarity by Score Range')
plt.suptitle('')  # 기본 제목 제거
plt.xlabel('Score')
plt.ylabel('Cosine Similarity')
plt.xticks(rotation=0)
plt.grid(True)

# 그래프 표시
plt.tight_layout()
plt.show()
