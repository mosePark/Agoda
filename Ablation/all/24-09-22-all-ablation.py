'''
temp parameter : {0.1, 0.7, 1.5}
 
similarity KS test
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
from sklearn.metrics.pairwise import cosine_similarity

#%% 

'''
사용자 함수 정의
'''

# 문자열을 벡터로 변환하는 함수

def string_to_vector(data):
    # 이미 리스트라면 변환하지 않음
    if isinstance(data, str):
        # 문자열을 리스트로 변환
        return np.array(ast.literal_eval(data))
    else:
        return np.array(data)

# 코사인 유사도를 계산하는 함수
def cosine_similarity(vec1, vec2):
    return 1 - cosine(np.array(vec1), np.array(vec2))


# 코사인 유사도 계산 함수
def compute_sim(ori_ebd, gen1_ebd, gen2_ebd, gen1_1_ebd, idx):
    # F 분포 계산 (ori_ebd와 gen1_ebd 사이의 코사인 유사도)
    F = [cosine_similarity(string_to_vector(ori), string_to_vector(gen1)) 
         for ori, gen1 in tqdm(zip(ori_ebd, gen1_ebd), total=len(idx), desc="Calculating F")]

    # G 분포 계산 (gen1_ebd와 gen2_ebd 사이의 코사인 유사도)
    G = [cosine_similarity(string_to_vector(gen1), string_to_vector(gen2)) 
         for gen1, gen2 in tqdm(zip(gen1_ebd, gen2_ebd), total=len(idx), desc="Calculating G")]

    # H 분포 계산 (gen1_ebd와 gen1_1_ebd 사이의 코사인 유사도)
    H = [cosine_similarity(string_to_vector(gen1), string_to_vector(gen1_1)) 
         for gen1, gen1_1 in tqdm(zip(gen1_ebd, gen1_1_ebd), total=len(idx), desc="Calculating H")]

    return F, G, H

# index 비율 샘플링
def sampling(dataframes, sample_fraction):

    num_rows = len(dataframes)
    idx_range = np.arange(0, num_rows)

    sampled_idx = np.random.choice(idx_range, size=int(num_rows * sample_fraction), replace=False)

    return sampled_idx



#%%
'''
데이터 로드
'''
# os.chdir('C:/Users/mose/agoda/data/') # home
# os.chdir('C:/Users/UOS/Desktop/Agoda-Data/') # lab
# os.chdir('/home1/mose1103/agoda/LLM') # UBAI
os.chdir('D:/mose/data/ablation2') # D-drive


df_1 = pd.read_csv("df_1.csv", index_col=0, encoding='utf-8-sig') # 0.1-gen1-ebd
df_2 = pd.read_csv("df_2.csv", index_col=0, encoding='utf-8-sig') # 0.1-gen2-ebd
df_3 = pd.read_csv("df_3.csv", index_col=0, encoding='utf-8-sig') # 0.1-gen1-1-ebd
df_4 = pd.read_csv("df_4.csv", index_col=0, encoding='utf-8-sig') # 0.7-gen1-ebd
df_5 = pd.read_csv("df_5.csv", index_col=0, encoding='utf-8-sig') # 0.7-gen2-ebd
df_6 = pd.read_csv("df_6.csv", index_col=0, encoding='utf-8-sig') # 0.7-gen1-1-ebd
df_7 = pd.read_csv("df_7.csv", index_col=0, encoding='utf-8-sig') # 1.5-gen1-ebd
df_8 = pd.read_csv("df_8.csv", index_col=0, encoding='utf-8-sig') # 1.5-gen2-ebd
df_9 = pd.read_csv("df_9.csv", index_col=0, encoding='utf-8-sig') # 1.5-gen1-1-ebd

#%%
# df 7, 9번 hotel 결측 처리
# df_7 = df_7.dropna(subset=['Hotel']).reset_index(drop=True)
# df_9 = df_9.dropna(subset=['Hotel']).reset_index(drop=True)

#%%
'''
setting : 0.1
'''
# 각 벡터 데이터를 추출
ori_ebd = df_6['ori-ebd'].values
gen1_ebd = df_1['0.1-gen1-ebd'].values
gen2_ebd = df_2['0.1-gen2-ebd'].values
gen1_1_ebd = df_3['0.1-gen1-1-ebd'].values

# 전체 데이터를 사용하여 F, G, H 분포 계산 (미리 계산)
F_full, G_full, H_full = compute_sim(ori_ebd, gen1_ebd, gen2_ebd, gen1_1_ebd, range(len(df_1)))
#%%

# 시뮬레이션 반복 횟수
num_cycles = 100
x_vals_cdf = np.linspace(0, 1, 500)  # CDF를 그릴 x축 값 범위
x_vals_pdf = np.linspace(-1, 1, 500)  # PDF를 그릴 x축 값 범위

# 각 사이클에서 CDF와 PDF 값을 저장할 리스트
F_cdf_list = []
G_cdf_list = []
H_cdf_list = []

F_pdf_list = []
G_pdf_list = []
H_pdf_list = []

# 각 사이클에서 KS 테스트 결과 저장
ks_results_0_1 = []

#%%

# 시뮬레이션 루프
for cycle in tqdm(range(num_cycles), desc="Simulations"):

    # 샘플링된 인덱스 생성 (0.1 샘플링)
    idx = sampling(F_full, 0.1)

    # 샘플링된 F, G, H 데이터를 사용
    F = np.array(F_full)[idx]
    G = np.array(G_full)[idx]
    H = np.array(H_full)[idx]

    # CDF 계산
    F_values = np.sort(F)
    F_cdf = np.arange(1, len(F_values) + 1) / len(F_values)

    G_values = np.sort(G)
    G_cdf = np.arange(1, len(G_values) + 1) / len(G_values)

    H_values = np.sort(H)
    H_cdf = np.arange(1, len(H_values) + 1) / len(H_values)

    # PDF 계산
    F_kde = gaussian_kde(F)
    G_kde = gaussian_kde(G)
    H_kde = gaussian_kde(H)

    F_pdf = F_kde(x_vals_pdf)
    G_pdf = G_kde(x_vals_pdf)
    H_pdf = H_kde(x_vals_pdf)

    # 각 사이클의 CDF와 PDF 결과 저장
    F_cdf_list.append(F_cdf)
    G_cdf_list.append(G_cdf)
    H_cdf_list.append(H_cdf)

    F_pdf_list.append(F_pdf)
    G_pdf_list.append(G_pdf)
    H_pdf_list.append(H_pdf)

    # KS 테스트 (F와 G, F와 H, G와 H 비교)
    ks_F_G = ks_2samp(F, G)
    ks_F_H = ks_2samp(F, H)
    ks_G_H = ks_2samp(G, H)

    ks_results_0_1.append((ks_F_G, ks_F_H, ks_G_H))

#%%

# CDF 및 PDF 변동성 시각화

# CDF 평균 및 표준편차 계산
F_cdf_mean = np.mean(F_cdf_list, axis=0)
F_cdf_std = np.std(F_cdf_list, axis=0)

G_cdf_mean = np.mean(G_cdf_list, axis=0)
G_cdf_std = np.std(G_cdf_list, axis=0)

H_cdf_mean = np.mean(H_cdf_list, axis=0)
H_cdf_std = np.std(H_cdf_list, axis=0)

# PDF 평균 및 표준편차 계산
F_pdf_mean = np.mean(F_pdf_list, axis=0)
F_pdf_std = np.std(F_pdf_list, axis=0)

G_pdf_mean = np.mean(G_pdf_list, axis=0)
G_pdf_std = np.std(G_pdf_list, axis=0)

H_pdf_mean = np.mean(H_pdf_list, axis=0)
H_pdf_std = np.std(H_pdf_list, axis=0)

#%%
# 최종 CDF 시각화
plt.figure(figsize=(8, 6))

plt.plot(F_values, F_cdf_mean, label='F (Ori vs Gen1)', color='blue')
plt.fill_between(F_values, F_cdf_mean - F_cdf_std, F_cdf_mean + F_cdf_std, color='blue', alpha=0.3)

plt.plot(G_values, G_cdf_mean, label='G (Gen1 vs Gen2)', color='red')
plt.fill_between(G_values, G_cdf_mean - G_cdf_std, G_cdf_mean + G_cdf_std, color='red', alpha=0.3)

plt.plot(H_values, H_cdf_mean, label='H (Gen1 vs Gen1-1)', color='green')
plt.fill_between(H_values, H_cdf_mean - H_cdf_std, H_cdf_mean + H_cdf_std, color='green', alpha=0.3)

plt.title('CDF Comparison with Confidence Intervals')
plt.xlabel('Cosine Similarity')
plt.ylabel('Cumulative Distribution Function (CDF)')
plt.legend()
plt.grid(True)

plt.savefig("0.1-cdf.png", dpi=1000)

plt.show()

#%%

# 최종 PDF 시각화
plt.figure(figsize=(8, 6))

plt.plot(x_vals_pdf, F_pdf_mean, label='F (Ori vs Gen1)', color='blue')
plt.fill_between(x_vals_pdf, F_pdf_mean - F_pdf_std, F_pdf_mean + F_pdf_std, color='blue', alpha=0.3)

plt.plot(x_vals_pdf, G_pdf_mean, label='G (Gen1 vs Gen2)', color='red')
plt.fill_between(x_vals_pdf, G_pdf_mean - G_pdf_std, G_pdf_mean + G_pdf_std, color='red', alpha=0.3)

plt.plot(x_vals_pdf, H_pdf_mean, label='H (Gen1 vs Gen1-1)', color='green')
plt.fill_between(x_vals_pdf, H_pdf_mean - H_pdf_std, H_pdf_mean + H_pdf_std, color='green', alpha=0.3)

plt.title('PDF Comparison with Confidence Intervals')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

plt.savefig("0.1-pdf.png", dpi=1000)

plt.show()


#%%

# KS 테스트 결과 요약
ks_F_G_stats = [result[0].statistic for result in ks_results_0_1]
ks_F_H_stats = [result[1].statistic for result in ks_results_0_1]
ks_G_H_stats = [result[2].statistic for result in ks_results_0_1]

plt.figure(figsize=(8, 6))
plt.hist(ks_F_G_stats, bins=20, alpha=0.5, label='KS F vs G')
plt.hist(ks_F_H_stats, bins=20, alpha=0.5, label='KS F vs H')
plt.hist(ks_G_H_stats, bins=20, alpha=0.5, label='KS G vs H')

plt.title('KS Test Statistics Distribution over 100 Simulations')
plt.xlabel('KS Statistic')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

plt.savefig("0.1-kstest.png", dpi=1000)

plt.show()


# %%

'''
0.7
'''

# 각 벡터 데이터를 추출
# ori_ebd = df_6['ori-ebd'].values
gen1_ebd = df_4['0.7-gen1-ebd'].values
gen2_ebd = df_5['0.7-gen2-ebd'].values
gen1_1_ebd = df_6['0.7-gen1-1-ebd'].values

# 전체 데이터를 사용하여 F, G, H 분포 계산 (미리 계산)
F_full_0_7, G_full_0_7, H_full_0_7 = compute_sim(ori_ebd, gen1_ebd, gen2_ebd, gen1_1_ebd, range(len(df_1)))

# 시뮬레이션 반복 횟수
num_cycles = 100
x_vals_cdf = np.linspace(0, 1, 500)  # CDF를 그릴 x축 값 범위
x_vals_pdf = np.linspace(-1, 1, 500)  # PDF를 그릴 x축 값 범위

# 각 사이클에서 CDF와 PDF 값을 저장할 리스트
F_cdf_list = []
G_cdf_list = []
H_cdf_list = []

F_pdf_list = []
G_pdf_list = []
H_pdf_list = []

# 각 사이클에서 KS 테스트 결과 저장
ks_results_0_7 = []


# 시뮬레이션 루프
for cycle in tqdm(range(num_cycles), desc="Simulations"):

    # 샘플링된 인덱스 생성 (0.1 샘플링)
    idx = sampling(F_full_0_7, 0.1)

    # 샘플링된 F, G, H 데이터를 사용
    F = np.array(F_full_0_7)[idx]
    G = np.array(G_full_0_7)[idx]
    H = np.array(H_full_0_7)[idx]

    # CDF 계산
    F_values = np.sort(F)
    F_cdf = np.arange(1, len(F_values) + 1) / len(F_values)

    G_values = np.sort(G)
    G_cdf = np.arange(1, len(G_values) + 1) / len(G_values)

    H_values = np.sort(H)
    H_cdf = np.arange(1, len(H_values) + 1) / len(H_values)

    # PDF 계산
    F_kde = gaussian_kde(F)
    G_kde = gaussian_kde(G)
    H_kde = gaussian_kde(H)

    F_pdf = F_kde(x_vals_pdf)
    G_pdf = G_kde(x_vals_pdf)
    H_pdf = H_kde(x_vals_pdf)

    # 각 사이클의 CDF와 PDF 결과 저장
    F_cdf_list.append(F_cdf)
    G_cdf_list.append(G_cdf)
    H_cdf_list.append(H_cdf)

    F_pdf_list.append(F_pdf)
    G_pdf_list.append(G_pdf)
    H_pdf_list.append(H_pdf)

    # KS 테스트 (F와 G, F와 H, G와 H 비교)
    ks_F_G = ks_2samp(F, G)
    ks_F_H = ks_2samp(F, H)
    ks_G_H = ks_2samp(G, H)

    ks_results_0_7.append((ks_F_G, ks_F_H, ks_G_H))


# CDF 및 PDF 변동성 시각화

# CDF 평균 및 표준편차 계산
F_cdf_mean = np.mean(F_cdf_list, axis=0)
F_cdf_std = np.std(F_cdf_list, axis=0)

G_cdf_mean = np.mean(G_cdf_list, axis=0)
G_cdf_std = np.std(G_cdf_list, axis=0)

H_cdf_mean = np.mean(H_cdf_list, axis=0)
H_cdf_std = np.std(H_cdf_list, axis=0)

# PDF 평균 및 표준편차 계산
F_pdf_mean = np.mean(F_pdf_list, axis=0)
F_pdf_std = np.std(F_pdf_list, axis=0)

G_pdf_mean = np.mean(G_pdf_list, axis=0)
G_pdf_std = np.std(G_pdf_list, axis=0)

H_pdf_mean = np.mean(H_pdf_list, axis=0)
H_pdf_std = np.std(H_pdf_list, axis=0)


# 최종 CDF 시각화
plt.figure(figsize=(8, 6))

plt.plot(F_values, F_cdf_mean, label='F (Ori vs Gen1)', color='blue')
plt.fill_between(F_values, F_cdf_mean - F_cdf_std, F_cdf_mean + F_cdf_std, color='blue', alpha=0.3)

plt.plot(G_values, G_cdf_mean, label='G (Gen1 vs Gen2)', color='red')
plt.fill_between(G_values, G_cdf_mean - G_cdf_std, G_cdf_mean + G_cdf_std, color='red', alpha=0.3)

plt.plot(H_values, H_cdf_mean, label='H (Gen1 vs Gen1-1)', color='green')
plt.fill_between(H_values, H_cdf_mean - H_cdf_std, H_cdf_mean + H_cdf_std, color='green', alpha=0.3)

plt.title('CDF Comparison with Confidence Intervals (0.7 Setting)')
plt.xlabel('Cosine Similarity')
plt.ylabel('Cumulative Distribution Function (CDF)')
plt.legend()
plt.grid(True)

plt.savefig("0.7-cdf.png", dpi=1000)

plt.show()


# 최종 PDF 시각화
plt.figure(figsize=(8, 6))

plt.plot(x_vals_pdf, F_pdf_mean, label='F (Ori vs Gen1)', color='blue')
plt.fill_between(x_vals_pdf, F_pdf_mean - F_pdf_std, F_pdf_mean + F_pdf_std, color='blue', alpha=0.3)

plt.plot(x_vals_pdf, G_pdf_mean, label='G (Gen1 vs Gen2)', color='red')
plt.fill_between(x_vals_pdf, G_pdf_mean - G_pdf_std, G_pdf_mean + G_pdf_std, color='red', alpha=0.3)

plt.plot(x_vals_pdf, H_pdf_mean, label='H (Gen1 vs Gen1-1)', color='green')
plt.fill_between(x_vals_pdf, H_pdf_mean - H_pdf_std, H_pdf_mean + H_pdf_std, color='green', alpha=0.3)

plt.title('PDF Comparison with Confidence Intervals (0.7 Setting)')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

plt.savefig("0.7-pdf.png", dpi=1000)

plt.show()


# KS 테스트 결과 요약
ks_F_G_stats = [result[0].statistic for result in ks_results_0_7]
ks_F_H_stats = [result[1].statistic for result in ks_results_0_7]
ks_G_H_stats = [result[2].statistic for result in ks_results_0_7]

plt.figure(figsize=(8, 6))
plt.hist(ks_F_G_stats, bins=20, alpha=0.5, label='KS F vs G')
plt.hist(ks_F_H_stats, bins=20, alpha=0.5, label='KS F vs H')
plt.hist(ks_G_H_stats, bins=20, alpha=0.5, label='KS G vs H')

plt.title('KS Test Statistics Distribution over 100 Simulations (0.7 Setting)')
plt.xlabel('KS Statistic')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

plt.savefig("0.7-kstest.png", dpi=1000)

plt.show()

#%%

'''
1.5
'''

# 각 벡터 데이터를 추출
# ori_ebd = df_6['ori-ebd'].values
gen1_ebd = df_7['1.5-gen1-ebd'].values
gen2_ebd = df_8['1.5-gen2-ebd'].values
gen1_1_ebd = df_9['1.5-gen1-1-ebd'].values

# 전체 데이터를 사용하여 F, G, H 분포 계산 (미리 계산)
F_full_1_5, G_full_1_5, H_full_1_5 = compute_sim(ori_ebd, gen1_ebd, gen2_ebd, gen1_1_ebd, range(len(df_1)))

# 시뮬레이션 반복 횟수
num_cycles = 100
x_vals_cdf = np.linspace(0, 1, 500)  # CDF를 그릴 x축 값 범위
x_vals_pdf = np.linspace(-1, 1, 500)  # PDF를 그릴 x축 값 범위

# 각 사이클에서 CDF와 PDF 값을 저장할 리스트
F_cdf_list = []
G_cdf_list = []
H_cdf_list = []

F_pdf_list = []
G_pdf_list = []
H_pdf_list = []

# 각 사이클에서 KS 테스트 결과 저장
ks_results_1_5 = []


# 시뮬레이션 루프
for cycle in tqdm(range(num_cycles), desc="Simulations"):

    # 샘플링된 인덱스 생성 (0.1 샘플링)
    idx = sampling(F_full_1_5, 0.1)

    # 샘플링된 F, G, H 데이터를 사용
    F = np.array(F_full_1_5)[idx]
    G = np.array(G_full_1_5)[idx]
    H = np.array(H_full_1_5)[idx]

    # CDF 계산
    F_values = np.sort(F)
    F_cdf = np.arange(1, len(F_values) + 1) / len(F_values)

    G_values = np.sort(G)
    G_cdf = np.arange(1, len(G_values) + 1) / len(G_values)

    H_values = np.sort(H)
    H_cdf = np.arange(1, len(H_values) + 1) / len(H_values)

    # PDF 계산
    F_kde = gaussian_kde(F)
    G_kde = gaussian_kde(G)
    H_kde = gaussian_kde(H)

    F_pdf = F_kde(x_vals_pdf)
    G_pdf = G_kde(x_vals_pdf)
    H_pdf = H_kde(x_vals_pdf)

    # 각 사이클의 CDF와 PDF 결과 저장
    F_cdf_list.append(F_cdf)
    G_cdf_list.append(G_cdf)
    H_cdf_list.append(H_cdf)

    F_pdf_list.append(F_pdf)
    G_pdf_list.append(G_pdf)
    H_pdf_list.append(H_pdf)

    # KS 테스트 (F와 G, F와 H, G와 H 비교)
    ks_F_G = ks_2samp(F, G)
    ks_F_H = ks_2samp(F, H)
    ks_G_H = ks_2samp(G, H)

    ks_results_1_5.append((ks_F_G, ks_F_H, ks_G_H))


# CDF 및 PDF 변동성 시각화

# CDF 평균 및 표준편차 계산
F_cdf_mean = np.mean(F_cdf_list, axis=0)
F_cdf_std = np.std(F_cdf_list, axis=0)

G_cdf_mean = np.mean(G_cdf_list, axis=0)
G_cdf_std = np.std(G_cdf_list, axis=0)

H_cdf_mean = np.mean(H_cdf_list, axis=0)
H_cdf_std = np.std(H_cdf_list, axis=0)

# PDF 평균 및 표준편차 계산
F_pdf_mean = np.mean(F_pdf_list, axis=0)
F_pdf_std = np.std(F_pdf_list, axis=0)

G_pdf_mean = np.mean(G_pdf_list, axis=0)
G_pdf_std = np.std(G_pdf_list, axis=0)

H_pdf_mean = np.mean(H_pdf_list, axis=0)
H_pdf_std = np.std(H_pdf_list, axis=0)


# 최종 CDF 시각화
plt.figure(figsize=(8, 6))

plt.plot(F_values, F_cdf_mean, label='F (Ori vs Gen1)', color='blue')
plt.fill_between(F_values, F_cdf_mean - F_cdf_std, F_cdf_mean + F_cdf_std, color='blue', alpha=0.3)

plt.plot(G_values, G_cdf_mean, label='G (Gen1 vs Gen2)', color='red')
plt.fill_between(G_values, G_cdf_mean - G_cdf_std, G_cdf_mean + G_cdf_std, color='red', alpha=0.3)

plt.plot(H_values, H_cdf_mean, label='H (Gen1 vs Gen1-1)', color='green')
plt.fill_between(H_values, H_cdf_mean - H_cdf_std, H_cdf_mean + H_cdf_std, color='green', alpha=0.3)

plt.title('CDF Comparison with Confidence Intervals (1.5 Setting)')
plt.xlabel('Cosine Similarity')
plt.ylabel('Cumulative Distribution Function (CDF)')
plt.legend()
plt.grid(True)

plt.savefig("1.5-cdf.png", dpi=1000)

plt.show()


# 최종 PDF 시각화
plt.figure(figsize=(8, 6))

plt.plot(x_vals_pdf, F_pdf_mean, label='F (Ori vs Gen1)', color='blue')
plt.fill_between(x_vals_pdf, F_pdf_mean - F_pdf_std, F_pdf_mean + F_pdf_std, color='blue', alpha=0.3)

plt.plot(x_vals_pdf, G_pdf_mean, label='G (Gen1 vs Gen2)', color='red')
plt.fill_between(x_vals_pdf, G_pdf_mean - G_pdf_std, G_pdf_mean + G_pdf_std, color='red', alpha=0.3)

plt.plot(x_vals_pdf, H_pdf_mean, label='H (Gen1 vs Gen1-1)', color='green')
plt.fill_between(x_vals_pdf, H_pdf_mean - H_pdf_std, H_pdf_mean + H_pdf_std, color='green', alpha=0.3)

plt.title('PDF Comparison with Confidence Intervals (1.5 Setting)')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

plt.savefig("1.5-pdf.png", dpi=1000)

plt.show()

# KS 테스트 결과 요약
ks_F_G_stats = [result[0].statistic for result in ks_results_1_5]
ks_F_H_stats = [result[1].statistic for result in ks_results_1_5]
ks_G_H_stats = [result[2].statistic for result in ks_results_1_5]

plt.figure(figsize=(8, 6))
plt.hist(ks_F_G_stats, bins=20, alpha=0.5, label='KS F vs G')
plt.hist(ks_F_H_stats, bins=20, alpha=0.5, label='KS F vs H')
plt.hist(ks_G_H_stats, bins=20, alpha=0.5, label='KS G vs H')

plt.title('KS Test Statistics Distribution over 100 Simulations (1.5 Setting)')
plt.xlabel('KS Statistic')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

plt.savefig("1.5-kstest.png", dpi=1000)

plt.show()

#%%

# 유사도 데이터를 DataFrame으로 변환
F_full_1_5_df = pd.DataFrame(F_full_1_5, columns=["F_full_1_5"])
G_full_1_5_df = pd.DataFrame(G_full_1_5, columns=["G_full_1_5"])
H_full_1_5_df = pd.DataFrame(H_full_1_5, columns=["H_full_1_5"])

F_full_0_7_df = pd.DataFrame(F_full_0_7, columns=["F_full_0_7"])
G_full_0_7_df = pd.DataFrame(G_full_0_7, columns=["G_full_0_7"])
H_full_0_7_df = pd.DataFrame(H_full_0_7, columns=["H_full_0_7"])

F_full_df = pd.DataFrame(F_full, columns=["F_full"])
G_full_df = pd.DataFrame(G_full, columns=["G_full"])
H_full_df = pd.DataFrame(H_full, columns=["H_full"])

# 각각 CSV 파일로 저장
F_full_1_5_df.to_csv("F_full_1_5.csv", index=False, encoding='utf-8-sig')
G_full_1_5_df.to_csv("G_full_1_5.csv", index=False, encoding='utf-8-sig')
H_full_1_5_df.to_csv("H_full_1_5.csv", index=False, encoding='utf-8-sig')

F_full_0_7_df.to_csv("F_full_0_7.csv", index=False, encoding='utf-8-sig')
G_full_0_7_df.to_csv("G_full_0_7.csv", index=False, encoding='utf-8-sig')
H_full_0_7_df.to_csv("H_full_0_7.csv", index=False, encoding='utf-8-sig')

F_full_df.to_csv("F_full.csv", index=False, encoding='utf-8-sig')
G_full_df.to_csv("G_full.csv", index=False, encoding='utf-8-sig')
H_full_df.to_csv("H_full.csv", index=False, encoding='utf-8-sig')
# %%
