'''
2세대 텍스트는 1세대 텍스트의 문맥을 잃는 경향이 있다.
- 데이터를 구체적으로 들여다보기
- 위 사실을 어떻게 표현할 수 있는지?
'''

import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
'''
데이터 로드
'''
# os.chdir('C:/Users/mose/agoda/data/') # home
# os.chdir('C:/Users/UOS/Desktop/Agoda-Data/ablation2') # lab
os.chdir('D:/Agoda-Data/ablation2/') # Lab, D-drive
# os.chdir('/home1/mose1103/agoda/LLM') # UBAI
# os.chdir('E:/mose/data/ablation2') # D-drive

df = pd.read_csv("df_no_ebd.csv", encoding='utf-8-sig')

#%%

'''

EDA : gen1-gen1-1(similarity distribution H) 사이의 문맥이 정말 잃는지 확인해볼 것

'''

#%% F 0.7 hist

plt.figure(figsize=(8, 6))

# 히스토그램 그리기
plt.hist(df['F_0_7'], bins=10)

# 그래프 설정
plt.title('Histogram of ori-gen1 similarity')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')

# x축을 0.1 단위로 설정
plt.xticks(np.arange(0, 1.1, 0.1))

# 그리드와 그래프 표시
plt.grid(True)
plt.show()


#%% H_0_7

plt.figure(figsize=(8, 6))

# 히스토그램 그리기
plt.hist(df['H_0_7'], bins=10)

# 그래프 설정
plt.title('Histogram of ori-gen1 similarity')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')

# x축을 0.1 단위로 설정
plt.xticks(np.arange(0, 1.1, 0.1))

# 그리드와 그래프 표시
plt.grid(True)
plt.show()

#%% F_0_7과 H_0_7

# F_0_7과 H_0_7을 동시에 그리는 히스토그램
plt.figure(figsize=(8, 6))

# F_0_7 히스토그램
plt.hist(df['F_0_7'], bins=10, alpha=0.5, label='F_0_7', color='blue', density=True)

# H_0_7 히스토그램
plt.hist(df['H_0_7'], bins=10, alpha=0.5, label='H_0_7', color='green', density=True)

# 그래프 설정
plt.title('Histogram of F_0_7 and H_0_7')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')

# x축을 0.1 단위로 설정
plt.xticks(np.arange(0, 1.1, 0.1))

# 범례와 그리드 설정
plt.legend(loc='upper right')
plt.grid(True)

# 그래프 표시
plt.show()

#%%


# 히스토그램 그리기 - G_0_7, F_0_7, H_0_7 비교

plt.hist(df['F_0_7'], bins=10, alpha=0.5, label='F_0_7', color='red', density=True)
plt.hist(df['H_0_7'], bins=10, alpha=0.5, label='H_0_7', color='green', density=True)


# 그래프 설정
plt.title('Histogram for G_0_7, F_0_7, and H_0_7')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
# %%

plt.hist(df['H_0_1'], bins=10, alpha=0.5, label='H_0_1', color='red', density=True)
plt.hist(df['H_0_7'], bins=10, alpha=0.5, label='H_0_7', color='green', density=True)
plt.hist(df['H_1_5'], bins=10, alpha=0.5, label='H_1_5', color='blue', density=True)

# 그래프 설정
plt.title('Histogram for H_0_1, H_0_7, and H_1_5')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
# %%

df[df['F_0_7'] > df['H_0_7']].to_csv('gen1-1에서 문맥 더 잃는지.csv', index=False, encoding='utf-8-sig')

df[df['F_0_7'] <= df['H_0_7']]

#%%

df[df['F_0_1'] > df['H_0_1']].to_csv('t 0.1 gen1-1에서 문맥 더 잃는지.csv', index=False, encoding='utf-8-sig')

df[df['F_1_5'] > df['H_1_5']].to_csv('t 1.5 gen1-1에서 문맥 더 잃는지.csv', index=False, encoding='utf-8-sig')

#%%

# 1. 데이터 필터링
filtered_df = df[df['F_0_7'] <= df['H_0_7']].copy()

# 2. 두 유사도 값의 차이 계산
filtered_df['Difference'] = filtered_df['H_0_7'] - filtered_df['F_0_7']

# 3. 히스토그램 시각화
max_difference = filtered_df['Difference'].max()
bins = np.arange(0, max_difference + 0.1, 0.1)

plt.figure(figsize=(10, 6))
plt.hist(filtered_df['Difference'], bins=bins, edgecolor='black', color='skyblue')
plt.title('두 유사도 값의 차이 빈도 분포', fontsize=15)
plt.xlabel('유사도 차이 (H_0_7 - F_0_7)', fontsize=12)
plt.ylabel('빈도수', fontsize=12)
plt.xticks(bins, rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# %%

# 차이 계산 (절대값 제거)
df['Difference'] = df['F_0_7'] - df['H_0_7']

# 히스토그램 시각화
min_difference = df['Difference'].min()
max_difference = df['Difference'].max()

# bin 간격을 0.1로 설정하고, 범위에 맞게 np.arange를 수정
bins = np.arange(min_difference - 0.1, max_difference + 0.1, 0.1)

plt.figure(figsize=(10, 6))
plt.hist(df['Difference'], bins=bins, edgecolor='black', color='skyblue')
plt.title('Diff of H, F', fontsize=15)
plt.xlabel('F_0_7 - H_0_7', fontsize=12)
plt.ylabel('Counts', fontsize=12)
plt.xticks(bins, rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 0 포인트 라인 그리기
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)

plt.tight_layout()
plt.show()
