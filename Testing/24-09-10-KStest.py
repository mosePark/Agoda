#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import ast


from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp
from scipy.stats import cumfreq
from scipy.stats import gaussian_kde
#%%
# 문자열로 저장된 벡터를 리스트로 변환하는 함수
def string_to_vector(string):
    return np.array(ast.literal_eval(string))

# 코사인 유사도를 계산하는 함수
def cosine_similarity(vec1, vec2):
    return 1 - cosine(np.array(vec1), np.array(vec2))

#%%
'''
데이터 로드
'''

os.chdir('C:/Users/mose/agoda/data/all') # home
# os.chdir('C:/Users/UOS/Desktop/Agoda-Data/raw') # lab


df = pd.read_csv("gen1-gen2-ebd.csv", index_col='Unnamed: 0')

#%% 

# ori_ebd와 gen1_ebd 사이의 코사인 유사도 계산 (F 분포)
F = [cosine_similarity(string_to_vector(ori), string_to_vector(gen1)) 
     for ori, gen1 in tqdm(zip(df['ori_ebd'], df['gen1_ebd']), total=len(df), desc="Calculating F")]

# gen1_ebd와 gen2_ebd 사이의 코사인 유사도 계산 (G 분포)
G = [cosine_similarity(string_to_vector(gen1), string_to_vector(gen2)) 
     for gen1, gen2 in tqdm(zip(df['gen1_ebd'], df['gen2_ebd']), total=len(df), desc="Calculating G")]

# Kolmogorov-Smirnov 검정 수행
ks_statistic, p_value = ks_2samp(F, G)

# 결과 출력
print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")
# %%

# F 분포의 누적 분포 함수(CDF)
F_values = np.sort(F)
F_cdf = np.arange(1, len(F_values) + 1) / len(F_values)

# G 분포의 누적 분포 함수(CDF)
G_values = np.sort(G)
G_cdf = np.arange(1, len(G_values) + 1) / len(G_values)

# 시각화
plt.figure(figsize=(8, 6))

# F와 G의 누적 분포 함수(CDF) 그리기
plt.plot(F_values, F_cdf, label='F (ori-gen)', color='blue')
plt.plot(G_values, G_cdf, label='G (gen-gen)', color='red')

# KS 통계량을 시각적으로 표시
ks_statistic, _ = ks_2samp(F, G)
max_diff_index = np.argmax(np.abs(F_cdf - G_cdf))
max_diff_value = F_values[max_diff_index]

plt.axvline(x=max_diff_value, color='green', linestyle='--', label=f'Max KS Stat = {ks_statistic:.3f}')

plt.title('CDF Comparison between F and G')
plt.xlabel('Cosine Similarity')
plt.ylabel('Cumulative Distribution Function (CDF)')
plt.legend()
plt.grid(True)
plt.show()
# %%

# F와 G의 PDF 계산을 위한 커널 밀도 추정
F_kde = gaussian_kde(F)
G_kde = gaussian_kde(G)

# F와 G의 PDF를 위한 x 값 생성
x_min = min(min(F), min(G))
x_max = max(max(F), max(G))
x_vals = np.linspace(x_min, x_max, 500)

# 시각화 (F와 G의 PDF를 동일한 그래프에 그림)
plt.figure(figsize=(8, 6))

# F의 PDF 그리기
plt.plot(x_vals, F_kde(x_vals), label='F (ori-gen)', color='blue')

# G의 PDF 그리기
plt.plot(x_vals, G_kde(x_vals), label='G (gen-gen)', color='red')

# 그래프 설정
plt.title('Probability Density Function (PDF) Comparison')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

# 그래프 보여주기
plt.show()
# %%
