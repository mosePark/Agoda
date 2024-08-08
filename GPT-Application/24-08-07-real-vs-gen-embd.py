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
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

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

# 각 유사도에 대해 MSE 계산
def calculate_mse_for_similarity(df, similarity_column):
    mse_by_bin = {}
    for i in range(len(bins) - 1):
        lower_bound = bins[i]
        upper_bound = bins[i + 1]
        bin_data = df[(df[similarity_column] >= lower_bound) & (df[similarity_column] < upper_bound)]
        if not bin_data.empty:
            mse = mean_squared_error(bin_data['Score'], bin_data['pred'])
            mse_by_bin[f'{lower_bound}-{upper_bound}'] = mse
    return mse_by_bin

#%%
os.chdir('C:/Users/mose/agoda/data/')


df1 = pd.read_csv("24-07-29-임베딩추가.csv", encoding='utf-8-sig')
df2 = pd.read_csv("24-07-31-r-임베딩추가.csv", encoding='utf-8-sig')
df3 = pd.read_csv("24-08-08-생성텍스트임베딩추가.csv", encoding='utf-8-sig')


#%%
len(df1), len(df2), len(df3)


#%%

# 각 데이터프레임의 중복 키 조합 확인
def check_duplicates(df, name):
    duplicated_rows = df.duplicated(subset=['hotel_name', 'Score', 'Title', 'Text', 'Date'], keep=False)
    duplicate_count = df[duplicated_rows].shape[0]
    print(f"Number of duplicated key combinations in {name}: {duplicate_count}")

check_duplicates(df1, 'df1')
check_duplicates(df2, 'df2')
check_duplicates(df3, 'df3')


#%%

# 1. df2와 df3 병합
df2_3_merged = pd.merge(df2, df3[['hotel_name', 'Score', 'Title', 'Text', 'Date', 'gen-embd', 'generated_review']],
                        on=['hotel_name', 'Score', 'Title', 'Text', 'Date'], how='left')

# 2. df2_3 병합 결과에서 중복 제거
df2_3_deduplicated = df2_3_merged.drop_duplicates(subset=['hotel_name', 'Score', 'Title', 'Text', 'Date'])

# 3. df1과 df2_3 병합 결과 병합
final_merged = pd.merge(df1, df2_3_deduplicated, on=['hotel_name', 'Score', 'Title', 'Text', 'Date'], how='left',
                        suffixes=('', '_drop'))

# 4. 불필요한 중복 칼럼 제거
final_merged = final_merged.loc[:, ~final_merged.columns.str.endswith('_drop')]

# 5. 최종 병합 결과에서 중복 제거
final_deduplicated_df = final_merged.drop_duplicates(subset=['hotel_name', 'Score', 'Title', 'Text', 'Date'])
# %%

# 결측치가 있는 행을 제거할 열 목록
cols_with_missing_values = ['Text', 'y_hat', 'pred', 'Reason', 'r-embd', 'gen-embd', 'generated_review']

# 결측치가 있는 행 제거
df_cleaned = final_deduplicated_df.dropna(subset=cols_with_missing_values)
df_cleaned

#%%

df_cleaned.isnull().sum()

df = df_cleaned


#%% 유사도 점수 계산

# 데이터프레임의 각 행에 대해 변환 함수 적용
df['embedding'] = df['embedding'].apply(convert_to_float_list)
df['r-embd'] = df['r-embd'].apply(convert_to_float_list)
df['gen-embd'] = df['gen-embd'].apply(convert_to_float_list)


embd = df['embedding'].to_list()
embd_r = df['r-embd'].to_list()
embd_g = df['gen-embd'].to_list()

# 코사인 유사도 계산
cosine_g_r = calculate_cosine_similarity_in_chunks(embd_g, embd_r, chunk_size=1000)
cosine_g_ = calculate_cosine_similarity_in_chunks(embd_g, embd, chunk_size=1000)
cosine_r_ = calculate_cosine_similarity_in_chunks(embd_r, embd, chunk_size=1000)

len(cosine_g_r), len(cosine_g_)

df['cosine_g_r'] = cosine_g_r
df['cosine_g_'] = cosine_g_
df['cosine_r_'] = cosine_r_

#%% 히스토그램

# 히스토그램을 서브플롯으로 그리기
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# 각 subplot에 대해 히스토그램 그리기
axes[0].hist(df['cosine_g_r'], bins=10, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_title('Histogram of Cosine_g_r')
axes[0].set_xlabel('Cosine Similarity')
axes[0].set_ylabel('Frequency')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

axes[1].hist(df['cosine_g_'], bins=10, alpha=0.7, color='green', edgecolor='black')
axes[1].set_title('Histogram of Cosine_g_')
axes[1].set_xlabel('Cosine Similarity')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

axes[2].hist(df['cosine_r_'], bins=10, alpha=0.7, color='red', edgecolor='black')
axes[2].set_title('Histogram of Cosine_r_')
axes[2].set_xlabel('Cosine Similarity')
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
# %%

# 코사인 유사도의 빈도를 0.1 단위로 나누기 위한 구간 생성
bins = np.arange(0, 1.1, 0.1)

# 서브플롯 생성
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# cosine_g_r 히스토그램
counts, bins, patches = axes[0].hist(df['cosine_g_r'], bins=bins, edgecolor='black', alpha=0.7)
for count, bin, patch in zip(counts, bins, patches):
    height = patch.get_height()
    axes[0].text(patch.get_x() + patch.get_width() / 2, height, f'{count / len(df):.2%}', 
                 ha='center', va='bottom')
axes[0].set_title('Histogram of Cosine_g_r')
axes[0].set_xlabel('Cosine Similarity')
axes[0].set_ylabel('Frequency')
axes[0].set_xticks(bins)
axes[0].grid(True)

# cosine_g_ 히스토그램
counts, bins, patches = axes[1].hist(df['cosine_g_'], bins=bins, edgecolor='black', alpha=0.7)
for count, bin, patch in zip(counts, bins, patches):
    height = patch.get_height()
    axes[1].text(patch.get_x() + patch.get_width() / 2, height, f'{count / len(df):.2%}', 
                 ha='center', va='bottom')
axes[1].set_title('Histogram of Cosine_g_')
axes[1].set_xlabel('Cosine Similarity')
axes[1].set_xticks(bins)
axes[1].grid(True)

# cosine_r_ 히스토그램
counts, bins, patches = axes[2].hist(df['cosine_r_'], bins=bins, edgecolor='black', alpha=0.7)
for count, bin, patch in zip(counts, bins, patches):
    height = patch.get_height()
    axes[2].text(patch.get_x() + patch.get_width() / 2, height, f'{count / len(df):.2%}', 
                 ha='center', va='bottom')
axes[2].set_title('Histogram of Cosine_r_')
axes[2].set_xlabel('Cosine Similarity')
axes[2].set_xticks(bins)
axes[2].grid(True)

plt.tight_layout()
plt.show()
# %%

# 구간 설정
bins = np.arange(0, 1.1, 0.1)
bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins) - 1)]

# 각 유사도에 대해 MSE 계산
def calculate_mse_for_similarity(df, similarity_column):
    mse_by_bin = {}
    for i in range(len(bins) - 1):
        lower_bound = bins[i]
        upper_bound = bins[i + 1]
        bin_data = df[(df[similarity_column] >= lower_bound) & (df[similarity_column] < upper_bound)]
        if not bin_data.empty:
            mse = mean_squared_error(bin_data['Score'], bin_data['pred'])
            mse_by_bin[bin_labels[i]] = mse
    return mse_by_bin

mse_g_r = calculate_mse_for_similarity(df, 'cosine_g_r')
mse_g_ = calculate_mse_for_similarity(df, 'cosine_g_')
mse_r_ = calculate_mse_for_similarity(df, 'cosine_r_')

# 서브플롯 그리기
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# 유사도별 꺾은선 그래프 그리기
def plot_mse(ax, mse_data, title):
    mse_values = [mse_data[bin_label] if bin_label in mse_data else 0 for bin_label in bin_labels]
    x = range(len(bin_labels))
    ax.plot(x, mse_values, marker='o', linestyle='-', color='b')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.set_title(title)
    ax.set_xlabel('Cosine Similarity Range')
    ax.set_ylabel('MSE')
    ax.grid(True)

plot_mse(axes[0], mse_g_r, 'MSE by Cosine Similarity Range (cosine_g_r)')
plot_mse(axes[1], mse_g_, 'MSE by Cosine Similarity Range (cosine_g_)')
plot_mse(axes[2], mse_r_, 'MSE by Cosine Similarity Range (cosine_r_)')

plt.tight_layout()
plt.show()
# %%

# cosine_g_ 값이 0.3 미만인 행 필터링
filtered_df = df[df['cosine_g_'] < 0.3]

# 'Text'와 'generated_review' 열만 선택하여 출력
filtered_df[['Text', 'generated_review']]

filtered_df.index
# %%

# filtered_df.to_excel("생성vs실제유사도낮은거.xlsx")

#%%

filtered_df['Text']

filtered_df['Text'].apply(len).mean()
