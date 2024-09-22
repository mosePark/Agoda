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

#%% 

'''
사용자 함수 정의
'''

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
os.chdir('D:/mose/data/ablation') # D-drive


df_1 = pd.read_csv("0.1-gen1-ebd.csv", index_col=0, encoding='utf-8-sig')
df_2 = pd.read_csv("0.1-gen2-ebd.csv", index_col=0, encoding='utf-8-sig')
df_3 = pd.read_csv("0.1-gen1-1-ebd.csv", index_col=0, encoding='utf-8-sig')
df_4 = pd.read_csv("0.7-gen1-ebd.csv", index_col=0, encoding='utf-8-sig')
df_5 = pd.read_csv("0.7-gen2-ebd.csv", index_col=0, encoding='utf-8-sig')
df_6 = pd.read_csv("0.7-gen1-1-ebd.csv", index_col=0, encoding='utf-8-sig')
df_7 = pd.read_csv("1.5-gen1-ebd.csv", index_col=0, encoding='utf-8-sig')
df_8 = pd.read_csv("1.5-gen2-ebd.csv", index_col=0, encoding='utf-8-sig')
df_9 = pd.read_csv("1.5-gen1-1-ebd.csv", index_col=0, encoding='utf-8-sig')


#%%

'''
0.1
'''

# 0.1 세팅
ori_ebd_0_1 = df_6_sampled['ori_ebd'].values
gen1_ebd_0_1 = df_1_sampled['0.1-gen1-ebd'].values
gen2_ebd_0_1 = df_2_sampled['0.1-gen2-ebd'].values
gen1_1_ebd_0_1 = df_3_sampled['0.1-gen1-1-ebd'].values

F_0_1 = [cosine_similarity(string_to_vector(ori), string_to_vector(gen1)) 
         for ori, gen1 in tqdm(zip(ori_ebd_0_1, gen1_ebd_0_1), total=len(ori_ebd_0_1), desc="Calculating F")]

G_0_1 = [cosine_similarity(string_to_vector(gen1), string_to_vector(gen2)) 
         for gen1, gen2 in tqdm(zip(gen1_ebd_0_1, gen2_ebd_0_1), total=len(gen1_ebd_0_1), desc="Calculating G")]

H_0_1 = [cosine_similarity(string_to_vector(gen1), string_to_vector(gen1_1)) 
         for gen1, gen1_1 in tqdm(zip(gen1_ebd_0_1, gen1_1_ebd_0_1), total=len(gen1_ebd_0_1), desc="Calculating H")]

# KS 테스트 수행 (0.1 세팅)
ks_F_G_0_1 = ks_2samp(F_0_1, G_0_1)
ks_G_H_0_1 = ks_2samp(G_0_1, H_0_1)
ks_F_H_0_1 = ks_2samp(F_0_1, H_0_1)

#%% 
'''
0.7
'''
# 0.7 세팅
ori_ebd_0_7 = df_6_sampled['ori_ebd'].values
gen1_ebd_0_7 = df_4_sampled['0.7-gen1-ebd'].values
gen2_ebd_0_7 = df_5_sampled['0.7-gen2-ebd'].values
gen1_1_ebd_0_7 = df_6_sampled['0.7-gen1-1-ebd'].values

F_0_7 = [cosine_similarity(string_to_vector(ori), string_to_vector(gen1)) 
         for ori, gen1 in tqdm(zip(ori_ebd_0_7, gen1_ebd_0_7), total=len(ori_ebd_0_7), desc="Calculating F")]

G_0_7 = [cosine_similarity(string_to_vector(gen1), string_to_vector(gen2)) 
         for gen1, gen2 in tqdm(zip(gen1_ebd_0_7, gen2_ebd_0_7), total=len(gen1_ebd_0_7), desc="Calculating G")]

H_0_7 = [cosine_similarity(string_to_vector(gen1), string_to_vector(gen1_1)) 
         for gen1, gen1_1 in tqdm(zip(gen1_ebd_0_7, gen1_1_ebd_0_7), total=len(gen1_ebd_0_7), desc="Calculating H")]

# KS 테스트 수행 (0.7 세팅)
ks_F_G_0_7 = ks_2samp(F_0_7, G_0_7)
ks_G_H_0_7 = ks_2samp(G_0_7, H_0_7)
ks_F_H_0_7 = ks_2samp(F_0_7, H_0_7)


'''
1.5
'''

# 1.5 세팅
ori_ebd_1_5 = df_6_sampled['ori_ebd'].values
gen1_ebd_1_5 = df_7_sampled['1.5-gen1-ebd'].values
gen2_ebd_1_5 = df_8_sampled['1.5-gen2-ebd'].values
gen1_1_ebd_1_5 = df_9_sampled['1.5-gen1-1-ebd'].values

F_1_5 = [cosine_similarity(string_to_vector(ori), string_to_vector(gen1)) 
         for ori, gen1 in tqdm(zip(ori_ebd_1_5, gen1_ebd_1_5), total=len(ori_ebd_1_5), desc="Calculating F")]

G_1_5 = [cosine_similarity(string_to_vector(gen1), string_to_vector(gen2)) 
         for gen1, gen2 in tqdm(zip(gen1_ebd_1_5, gen2_ebd_1_5), total=len(gen1_ebd_1_5), desc="Calculating G")]

H_1_5 = [cosine_similarity(string_to_vector(gen1), string_to_vector(gen1_1)) 
         for gen1, gen1_1 in tqdm(zip(gen1_ebd_1_5, gen1_1_ebd_1_5), total=len(gen1_ebd_1_5), desc="Calculating H")]

# KS 테스트 수행 (1.5 세팅)
ks_F_G_1_5 = ks_2samp(F_1_5, G_1_5)
ks_G_H_1_5 = ks_2samp(G_1_5, H_1_5)
ks_F_H_1_5 = ks_2samp(F_1_5, H_1_5)

# 검정 통계량과 유의확률을 출력
print(f"F=G에 대한 KS 테스트 결과 (0.1 세팅): 검정 통계량 = {ks_F_G_0_1.statistic:.4f}, 유의확률 = {ks_F_G_0_1.pvalue:.4f}")
print(f"G=H에 대한 KS 테스트 결과 (0.1 세팅): 검정 통계량 = {ks_G_H_0_1.statistic:.4f}, 유의확률 = {ks_G_H_0_1.pvalue:.4f}")
print(f"F=H에 대한 KS 테스트 결과 (0.1 세팅): 검정 통계량 = {ks_F_H_0_1.statistic:.4f}, 유의확률 = {ks_F_H_0_1.pvalue:.4f}")

print(f"F=G에 대한 KS 테스트 결과 (0.7 세팅): 검정 통계량 = {ks_F_G_0_7.statistic:.4f}, 유의확률 = {ks_F_G_0_7.pvalue:.4f}")
print(f"G=H에 대한 KS 테스트 결과 (0.7 세팅): 검정 통계량 = {ks_G_H_0_7.statistic:.4f}, 유의확률 = {ks_G_H_0_7.pvalue:.4f}")
print(f"F=H에 대한 KS 테스트 결과 (0.7 세팅): 검정 통계량 = {ks_F_H_0_7.statistic:.4f}, 유의확률 = {ks_F_H_0_7.pvalue:.4f}")

print(f"F=G에 대한 KS 테스트 결과 (1.5 세팅): 검정 통계량 = {ks_F_G_1_5.statistic:.4f}, 유의확률 = {ks_F_G_1_5.pvalue:.4f}")
print(f"G=H에 대한 KS 테스트 결과 (1.5 세팅): 검정 통계량 = {ks_G_H_1_5.statistic:.4f}, 유의확률 = {ks_G_H_1_5.pvalue:.4f}")
print(f"F=H에 대한 KS 테스트 결과 (1.5 세팅): 검정 통계량 = {ks_F_H_1_5.statistic:.4f}, 유의확률 = {ks_F_H_1_5.pvalue:.4f}")
