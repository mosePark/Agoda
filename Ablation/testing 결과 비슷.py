'''
tempearature parameter 별로 KS test
'''
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp, gaussian_kde


#%% 문자열로 저장된 벡터를 리스트로 변환하는 함수

# 문자열을 벡터로 변환하는 함수
def string_to_vector(string):
    try:
        return np.array(ast.literal_eval(string))
    except ValueError:
        return None  # 변환에 실패하면 None 반환

# 코사인 유사도를 계산하는 함수
def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return None  # 벡터 변환이 실패했을 경우 처리
    return 1 - cosine(vec1, vec2)

# Ori vs gen1, gen1 vs gen2, gen1 vs gen1-1 각각의 코사인 유사도를 계산하는 함수
def compute_similarities(ori_ebd, gen1_ebd, gen2_ebd, gen1_1_ebd):
    sim_ori_gen1 = [cosine_similarity(string_to_vector(ori), string_to_vector(gen1)) 
                    for ori, gen1 in tqdm(zip(ori_ebd, gen1_ebd), total=len(ori_ebd), desc="Ori vs Gen1")]

    sim_gen1_gen2 = [cosine_similarity(string_to_vector(gen1), string_to_vector(gen2)) 
                     for gen1, gen2 in tqdm(zip(gen1_ebd, gen2_ebd), total=len(gen1_ebd), desc="Gen1 vs Gen2")]

    sim_gen1_gen1_1 = [cosine_similarity(string_to_vector(gen1), string_to_vector(gen1_1)) 
                       for gen1, gen1_1 in tqdm(zip(gen1_ebd, gen1_1_ebd), total=len(gen1_ebd), desc="Gen1 vs Gen1-1")]

    return sim_ori_gen1, sim_gen1_gen2, sim_gen1_gen1_1

#%%
'''
데이터 로드
'''
os.chdir('C:/Users/mose/agoda/data/') # home
# os.chdir('C:/Users/UOS/Desktop/Agoda-Data/') # lab
# os.chdir('/home1/mose1103/agoda/LLM') # UBAI

df_1 = pd.read_csv("ebd/0.1-gen1-1-ebd.csv", index_col=0, encoding='utf-8-sig')
df_2 = pd.read_csv("ebd/0.1-gen2-ebd.csv", index_col=0, encoding='utf-8-sig')
df_7 =  pd.read_csv("ebd/0.1-gen1-ebd.csv", index_col=0, encoding='utf-8-sig')

df_3 = pd.read_csv("ebd/1.5-gen1-ebd.csv", index_col=0, encoding='utf-8-sig')
df_4 = pd.read_csv("ebd/1.5-gen1-1-ebd.csv", index_col=0, encoding='utf-8-sig')
df_5 = pd.read_csv("ebd/1.5-gen2-ebd.csv", index_col=0, encoding='utf-8-sig')
df_6 = pd.read_csv("0.7-all.csv", index_col=0, encoding='utf-8-sig')

#%%
'''
temp : 0.1
'''

# 각 DataFrame에서 벡터 데이터를 가져오기
ori_ebd_0_1 = df_6['ori_ebd'].values
gen1_ebd_0_1 = df_7['0.1-gen1-ebd'].values
gen2_ebd_0_1 = df_2['0.1-gen2-ebd'].values
gen1_1_ebd_0_1 = df_1['0.1-gen1-1-ebd'].values

# 코사인 유사도 계산 (0.1 세팅)
sim_ori_gen1_0_1, sim_gen1_gen2_0_1, sim_gen1_gen1_1_0_1 = compute_similarities(ori_ebd_0_1, gen1_ebd_0_1, gen2_ebd_0_1, gen1_1_ebd_0_1)

# None 값 조회
none_count_F_0_1 = sim_ori_gen1_0_1.count(None)
none_count_G_0_1 = sim_gen1_gen2_0_1.count(None)
none_count_H_0_1 = sim_gen1_gen1_1_0_1.count(None)

print(f"F (Ori vs Gen1)에 있는 None 값의 개수: {none_count_F_0_1}")
print(f"G (Gen1 vs Gen2)에 있는 None 값의 개수: {none_count_G_0_1}")
print(f"H (Gen1 vs Gen1-1)에 있는 None 값의 개수: {none_count_H_0_1}")

# None 값 제거 (0.1 세팅)
F_0_1 = [val for val in sim_ori_gen1_0_1 if val is not None]
G_0_1 = [val for val in sim_gen1_gen2_0_1 if val is not None]
H_0_1 = [val for val in sim_gen1_gen1_1_0_1 if val is not None]

# KS 테스트 수행 (0.1 세팅)
ks_F_G_0_1 = ks_2samp(F_0_1, G_0_1)
ks_G_H_0_1 = ks_2samp(G_0_1, H_0_1)
ks_F_H_0_1 = ks_2samp(F_0_1, H_0_1)

# 검정 통계량과 유의확률을 소수점 넷째 자리까지 출력 (0.1 세팅)
print(f"F=G에 대한 KS 테스트 결과 (0.1 세팅): 검정 통계량 = {ks_F_G_0_1.statistic:.4f}, 유의확률 = {ks_F_G_0_1.pvalue:.4f}")
print(f"G=H에 대한 KS 테스트 결과 (0.1 세팅): 검정 통계량 = {ks_G_H_0_1.statistic:.4f}, 유의확률 = {ks_G_H_0_1.pvalue:.4f}")
print(f"F=H에 대한 KS 테스트 결과 (0.1 세팅): 검정 통계량 = {ks_F_H_0_1.statistic:.4f}, 유의확률 = {ks_F_H_0_1.pvalue:.4f}")

#%%

# CDF 시각화
# F 분포의 누적 분포 함수(CDF)
F_values_0_1 = np.sort(F_0_1)
F_cdf_0_1 = np.arange(1, len(F_values_0_1) + 1) / len(F_values_0_1)

# G 분포의 누적 분포 함수(CDF)
G_values_0_1 = np.sort(G_0_1)
G_cdf_0_1 = np.arange(1, len(G_values_0_1) + 1) / len(G_values_0_1)

# H 분포의 누적 분포 함수(CDF)
H_values_0_1 = np.sort(H_0_1)
H_cdf_0_1 = np.arange(1, len(H_values_0_1) + 1) / len(H_values_0_1)

# 시각화 (CDF 비교)
plt.figure(figsize=(8, 6))

# F, G, H의 누적 분포 함수(CDF) 그리기
plt.plot(F_values_0_1, F_cdf_0_1, label='F (Ori vs Gen1)', color='blue')
plt.plot(G_values_0_1, G_cdf_0_1, label='G (Gen1 vs Gen2)', color='red')
plt.plot(H_values_0_1, H_cdf_0_1, label='H (Gen1 vs Gen1-1)', color='green')

# 최대 차이 계산 (KS 통계량)
ks_statistic_F_G_0_1, _ = ks_2samp(F_0_1, G_0_1)
ks_statistic_F_H_0_1, _ = ks_2samp(F_0_1, H_0_1)
ks_statistic_G_H_0_1, _ = ks_2samp(G_0_1, H_0_1)

# KS 최대 차이 시각적으로 표시
max_diff_index_F_G_0_1 = np.argmax(np.abs(F_cdf_0_1 - G_cdf_0_1))
max_diff_value_F_G_0_1 = F_values_0_1[max_diff_index_F_G_0_1]

plt.axvline(x=max_diff_value_F_G_0_1, color='purple', linestyle='--', label=f'Max KS F-G = {ks_statistic_F_G_0_1:.3f}')
plt.axvline(x=max_diff_value_F_G_0_1, color='orange', linestyle='--', label=f'Max KS F-H = {ks_statistic_F_H_0_1:.3f}')
plt.axvline(x=max_diff_value_F_G_0_1, color='gray', linestyle='--', label=f'Max KS G-H = {ks_statistic_G_H_0_1:.3f}')

plt.title('CDF Comparison between F, G, and H (0.1 세팅)')
plt.xlabel('Cosine Similarity')
plt.ylabel('Cumulative Distribution Function (CDF)')
plt.legend()
plt.grid(True)
plt.show()

# PDF 시각화
# F, G, H의 PDF 계산을 위한 커널 밀도 추정
F_kde_0_1 = gaussian_kde(F_0_1)
G_kde_0_1 = gaussian_kde(G_0_1)
H_kde_0_1 = gaussian_kde(H_0_1)

# F, G, H의 PDF를 위한 x 값 생성
x_min_0_1 = min(min(F_0_1), min(G_0_1), min(H_0_1))
x_max_0_1 = max(max(F_0_1), max(G_0_1), max(H_0_1))
x_vals_0_1 = np.linspace(x_min_0_1, x_max_0_1, 500)

# 시각화 (F, G, H의 PDF 비교)
plt.figure(figsize=(8, 6))

# F의 PDF 그리기
plt.plot(x_vals_0_1, F_kde_0_1(x_vals_0_1), label='F (Ori vs Gen1)', color='blue')

# G의 PDF 그리기
plt.plot(x_vals_0_1, G_kde_0_1(x_vals_0_1), label='G (Gen1 vs Gen2)', color='red')

# H의 PDF 그리기
plt.plot(x_vals_0_1, H_kde_0_1(x_vals_0_1), label='H (Gen1 vs Gen1-1)', color='green')

# 그래프 설정
plt.title('Probability Density Function (PDF) Comparison (0.1 세팅)')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

# 그래프 보여주기
plt.show()



#%%
'''
temp :1.5
'''
# 1.5 세팅에 대한 코사인 유사도 계산
gen1_5_ebd = df_3['1.5-gen1-ebd'].values
gen1_5_1_ebd = df_4['1.5-gen1-1-ebd'].values
gen1_5_2_ebd = df_5['1.5-gen2-ebd'].values

sim_1_5_gen1_gen2 = [cosine_similarity(string_to_vector(gen1_5), string_to_vector(gen2_5)) 
                     for gen1_5, gen2_5 in tqdm(zip(gen1_5_ebd, gen1_5_2_ebd), total=len(gen1_5_ebd), desc="1.5 Gen1 vs Gen2")]

sim_1_5_gen1_gen1_1 = [cosine_similarity(string_to_vector(gen1_5), string_to_vector(gen1_5_1)) 
                       for gen1_5, gen1_5_1 in tqdm(zip(gen1_5_ebd, gen1_5_1_ebd), total=len(gen1_5_ebd), desc="1.5 Gen1 vs Gen1-1")]

# None 값 조회
none_count_F = sim_ori_gen1.count(None)
none_count_G = sim_gen1_gen2.count(None)
none_count_H = sim_gen1_gen1_1.count(None)

print(f"F (Ori vs Gen1)에 있는 None 값의 개수: {none_count_F}")
print(f"G (Gen1 vs Gen2)에 있는 None 값의 개수: {none_count_G}")
print(f"H (Gen1 vs Gen1-1)에 있는 None 값의 개수: {none_count_H}")

# None 값 제거
F = [val for val in sim_ori_gen1 if val is not None]
G = [val for val in sim_gen1_gen2 if val is not None]
H = [val for val in sim_gen1_gen1_1 if val is not None]

# KS 테스트 수행
ks_F_G = ks_2samp(F, G)
ks_G_H = ks_2samp(G, H)
ks_F_H = ks_2samp(F, H)

# 검정 통계량과 유의확률을 소수점 넷째 자리까지 출력
print(f"F=G에 대한 KS 테스트 결과: 검정 통계량 = {ks_F_G.statistic:.4f}, 유의확률 = {ks_F_G.pvalue:.4f}")
print(f"G=H에 대한 KS 테스트 결과: 검정 통계량 = {ks_G_H.statistic:.4f}, 유의확률 = {ks_G_H.pvalue:.4f}")
print(f"F=H에 대한 KS 테스트 결과: 검정 통계량 = {ks_F_H.statistic:.4f}, 유의확률 = {ks_F_H.pvalue:.4f}")

#%%

# F (Ori vs Gen1), G (Gen1 vs Gen2), H (Gen1 vs Gen1-1)의 CDF 계산 및 시각화

# F 분포의 누적 분포 함수(CDF)
F_values = np.sort(F)
F_cdf = np.arange(1, len(F_values) + 1) / len(F_values)

# G 분포의 누적 분포 함수(CDF)
G_values = np.sort(G)
G_cdf = np.arange(1, len(G_values) + 1) / len(G_values)

# H 분포의 누적 분포 함수(CDF)
H_values = np.sort(H)
H_cdf = np.arange(1, len(H_values) + 1) / len(H_values)

# 시각화 (CDF 비교)
plt.figure(figsize=(8, 6))

# F, G, H의 누적 분포 함수(CDF) 그리기
plt.plot(F_values, F_cdf, label='F (Ori vs Gen1)', color='blue')
plt.plot(G_values, G_cdf, label='G (Gen1 vs Gen2)', color='red')
plt.plot(H_values, H_cdf, label='H (Gen1 vs Gen1-1)', color='green')

# KS 통계량 계산 및 시각적으로 표시
ks_statistic_F_G, _ = ks_2samp(F, G)
ks_statistic_F_H, _ = ks_2samp(F, H)
ks_statistic_G_H, _ = ks_2samp(G, H)

# 최대 차이 지점 계산
max_diff_index_F_G = np.argmax(np.abs(F_cdf - G_cdf))
max_diff_value_F_G = F_values[max_diff_index_F_G]

plt.axvline(x=max_diff_value_F_G, color='purple', linestyle='--', label=f'Max KS F-G = {ks_statistic_F_G:.3f}')
plt.axvline(x=max_diff_value_F_G, color='orange', linestyle='--', label=f'Max KS F-H = {ks_statistic_F_H:.3f}')
plt.axvline(x=max_diff_value_F_G, color='gray', linestyle='--', label=f'Max KS G-H = {ks_statistic_G_H:.3f}')

plt.title('CDF Comparison between F, G, and H, Temp : 1.5')
plt.xlabel('Cosine Similarity')
plt.ylabel('Cumulative Distribution Function (CDF)')
plt.legend()
plt.grid(True)
plt.show()

# F, G, H의 PDF 계산 및 시각화

# F, G, H의 PDF 계산을 위한 커널 밀도 추정
F_kde = gaussian_kde(F)
G_kde = gaussian_kde(G)
H_kde = gaussian_kde(H)

# F, G, H의 PDF를 위한 x 값 생성
x_min = min(min(F), min(G), min(H))
x_max = max(max(F), max(G), max(H))
x_vals = np.linspace(x_min, x_max, 500)

# 시각화 (F, G, H의 PDF 비교)
plt.figure(figsize=(8, 6))

# F의 PDF 그리기
plt.plot(x_vals, F_kde(x_vals), label='F (Ori vs Gen1)', color='blue')

# G의 PDF 그리기
plt.plot(x_vals, G_kde(x_vals), label='G (Gen1 vs Gen2)', color='red')

# H의 PDF 그리기
plt.plot(x_vals, H_kde(x_vals), label='H (Gen1 vs Gen1-1)', color='green')

# 그래프 설정
plt.title('Probability Density Function (PDF) Comparison, Temp : 1.5')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

# 그래프 보여주기
plt.show()
# %%
