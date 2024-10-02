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
