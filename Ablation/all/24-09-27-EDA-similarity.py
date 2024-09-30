'''
유사도값 별로 어떤 데이터인지 살펴보기
'''

import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sampling(df, col, low, high, size):
    """
    Parameters:
    - df (DataFrame): 데이터프레임
    - col (str): 유사도 값이 들어있는 컬럼명 (예: 'F_0_1', 'G_0_1' 등)
    - low (float): 값의 하한
    - high (float): 값의 상한
    - size (int): 추출할 샘플 수

    Returns:
    - sampled_indices (Index): 샘플링 데이터 인덱스
    """

    # 필터링
    filtered_data = df[(df[col] > low) & (df[col] < high)]
    
    # 필터링된 데이터가 요청한 샘플 크기보다 작으면, 필터링된 데이터의 크기로 샘플 크기 제한
    actual_size = min(len(filtered_data), size)
    
    # 필터링된 데이터에서 랜덤하게 인덱스 추출 (비복원추출)
    sampled_indices = np.random.choice(filtered_data.index, size=actual_size, replace=False)
    
    return sampled_indices

def cal_text_stats(df, col, low, high, text_col):
    """
    특정 범위 내에서 텍스트의 평균 길이와 평균 토큰 수를 계산하는 함수.
    
    Parameters:
    - df (DataFrame): 데이터프레임
    - col (str): 필터링할 기준이 되는 컬럼명 (예: 'F_0_1')
    - low (float): 필터링할 하한 값
    - high (float): 필터링할 상한 값
    - text_col (str): 텍스트 데이터가 들어있는 컬럼명 (예: 'Text', '0.1-gen1')
    
    Returns:
    - avg_length (float): 텍스트의 평균 길이
    - avg_token_count (float): 텍스트의 평균 토큰 수
    """
    # 필터링
    filtered_data = df[(df[col] > low) & (df[col] < high)][text_col].dropna()

    # 텍스트 평균 길이
    avg_length = filtered_data.apply(len).mean()
    
    # 텍스트 평균 토큰 수
    avg_token_count = filtered_data.apply(lambda x: len(x.split())).mean()

    return avg_length, avg_token_count


#%%
'''
데이터 로드
'''
# os.chdir('C:/Users/mose/agoda/data/') # home
# os.chdir('C:/Users/UOS/Desktop/Agoda-Data/ablation2') # lab
os.chdir('D:/Agoda-Data/ablation2/') # Lab, D-drive
# os.chdir('/home1/mose1103/agoda/LLM') # UBAI
# os.chdir('E:/mose/data/ablation2') # D-drive

df_1 = pd.read_csv("df_1.csv", encoding='utf-8-sig') # 0.1-gen1-ebd
df_2 = pd.read_csv("df_2.csv", encoding='utf-8-sig') # 0.1-gen2-ebd
df_3 = pd.read_csv("df_3.csv", encoding='utf-8-sig') # 0.1-gen1-1-ebd

df_4 = pd.read_csv("df_4.csv", encoding='utf-8-sig') # 0.7-gen1-ebd
df_5 = pd.read_csv("df_5.csv", encoding='utf-8-sig') # 0.7-gen2-ebd
df_6 = pd.read_csv("df_6.csv", encoding='utf-8-sig') # 0.7-gen1-1-ebd

df_7 = pd.read_csv("df_7.csv", encoding='utf-8-sig') # 1.5-gen1-ebd
df_8 = pd.read_csv("df_8.csv", encoding='utf-8-sig') # 1.5-gen2-ebd
df_9 = pd.read_csv("df_9.csv", encoding='utf-8-sig') # 1.5-gen1-1-ebd

F_full_1_5 = pd.read_csv("F_full_1_5.csv", encoding='utf-8-sig')
G_full_1_5 = pd.read_csv("G_full_1_5.csv", encoding='utf-8-sig')
H_full_1_5 = pd.read_csv("H_full_1_5.csv", encoding='utf-8-sig')

F_full_0_7 = pd.read_csv("F_full_0_7.csv", encoding='utf-8-sig')
G_full_0_7 = pd.read_csv("G_full_0_7.csv", encoding='utf-8-sig')
H_full_0_7 = pd.read_csv("H_full_0_7.csv", encoding='utf-8-sig')

F_full_0_1 = pd.read_csv("F_full.csv", encoding='utf-8-sig')
G_full_0_1 = pd.read_csv("G_full.csv", encoding='utf-8-sig')
H_full_0_1 = pd.read_csv("H_full.csv", encoding='utf-8-sig')

#%%
'''
데이터 분류

* 임베딩은 임베딩끼리, 정보는 정보끼리
'''

dataframes = [df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9]

# 모든 데이터프레임 변수 이름 추출
column_names = {f'df_{i+1}': df.columns for i, df in enumerate(dataframes)}

# 빈 데이터프레임 생성
df_no_ebd = pd.DataFrame()
df_with_ebd = pd.DataFrame()


# # 공통 변수 추가
# common_columns = ['Hotel', 'Score', 'Country', 'Traveler Type', 'Room Type',
#                   'Stay Duration', 'Title', 'Text', 'Date', 'Full']

# # 각각의 공통 변수들 df_1 기준으로 추가
# for col in common_columns:
#     df_no_ebd[col] = df_1[col]
#     df_with_ebd[col] = df_1[col]

# 공통 변수 추가

df_no_ebd['Hotel'] = df_1['Hotel']
df_no_ebd['Score'] = df_1['Score']
df_no_ebd['Country'] = df_1['Country']
df_no_ebd['Traveler_Type'] = df_1['Traveler Type']
df_no_ebd['Room_Type'] = df_1['Room Type']
df_no_ebd['Stay_Duration'] = df_1['Stay Duration']
df_no_ebd['Title'] = df_1['Title']
df_no_ebd['Text'] = df_1['Text']
df_no_ebd['Date'] = df_1['Date']
df_no_ebd['Full'] = df_1['Full']

df_with_ebd['Hotel'] = df_1['Hotel']
df_with_ebd['Score'] = df_1['Score']
df_with_ebd['Country'] = df_1['Country']
df_with_ebd['Traveler_Type'] = df_1['Traveler Type']
df_with_ebd['Room_Type'] = df_1['Room Type']
df_with_ebd['Stay_Duration'] = df_1['Stay Duration']
df_with_ebd['Title'] = df_1['Title']
df_with_ebd['Text'] = df_1['Text']
df_with_ebd['Date'] = df_1['Date']
df_with_ebd['Full'] = df_1['Full']

#%%

# 임베딩 데이터프레임

df_with_ebd['ori-ebd'] = df_6['ori-ebd']

df_with_ebd['0.1-gen1-ebd'] = df_1['0.1-gen1-ebd']
df_with_ebd['0.1-gen2-ebd'] = df_2['0.1-gen2-ebd']
df_with_ebd['0.1-gen1-1-ebd'] = df_3['0.1-gen1-1-ebd']

df_with_ebd['0.7-gen1-ebd'] = df_4['0.7-gen1-ebd']
df_with_ebd['0.7-gen2-ebd'] = df_5['0.7-gen2-ebd']
df_with_ebd['0.7-gen1-1-ebd'] = df_6['0.7-gen1-1-ebd']

df_with_ebd['1.5-gen1-ebd'] = df_7['1.5-gen1-ebd']
df_with_ebd['1.5-gen2-ebd'] = df_8['1.5-gen2-ebd']
df_with_ebd['1.5-gen1-1-ebd'] = df_9['1.5-gen1-1-ebd']



#%%

# 임베딩 이외 데이터프레임

df_no_ebd['0.1-gen1'] = df_1['generated_review']
df_no_ebd['0.1-gen2'] = df_2['generated_review_2']
df_no_ebd['0.1-gen1-1'] = df_3['new_generated_review']

df_no_ebd['0.7-gen1'] = df_4['generated_review']
df_no_ebd['0.7-gen2'] = df_5['generated_review_2']
df_no_ebd['0.7-gen1-1'] = df_6['new_generated_review']

df_no_ebd['1.5-gen1'] = df_7['generated_review']
df_no_ebd['1.5-gen2'] = df_8['generated_review_2']
df_no_ebd['1.5-gen1-1'] = df_9['new_generated_review']

# 유사도값 넣기
df_no_ebd['F_0_1'] = F_full_0_1.iloc[:, 0]
df_no_ebd['G_0_1'] = G_full_0_1.iloc[:, 0]
df_no_ebd['H_0_1'] = H_full_0_1.iloc[:, 0]

df_no_ebd['F_0_7'] = F_full_0_7.iloc[:, 0]
df_no_ebd['G_0_7'] = G_full_0_7.iloc[:, 0]
df_no_ebd['H_0_7'] = H_full_0_7.iloc[:, 0]

df_no_ebd['F_1_5'] = F_full_1_5.iloc[:, 0]
df_no_ebd['G_1_5'] = G_full_1_5.iloc[:, 0]
df_no_ebd['H_1_5'] = H_full_1_5.iloc[:, 0]


#%%

# zip 압축으로 저장 - python용
df_with_ebd.to_csv('df_with_ebd.zip', index=False, compression='zip', encoding='utf-8-sig')

# R에서는 그냥 df_1 ~ df_9로 사용하자.
# 파일저장
df_no_ebd.to_csv('df_no_ebd.csv', index=False, encoding='utf-8-sig')

#%%

# # 청크 단위로 데이터 읽기
# chunk_size = 1000
# chunks = []

# # zip 파일을 청크 단위로 읽으면서 진행 상황을 tqdm으로 표시
# with tqdm(total=100) as pbar:  # tqdm의 총 길이를 설정합니다 (전체 크기는 알 수 없으니 대략 설정)
#     for chunk in pd.read_csv('df_with_ebd.zip', compression='zip', encoding='utf-8-sig', chunksize=chunk_size):
#         chunks.append(chunk)
#         pbar.update(1)  # tqdm 바 업데이트

# # 청크를 합쳐서 하나의 데이터프레임으로 결합
# a = pd.concat(chunks, ignore_index=True)
#%%

df_no_ebd.isnull().sum()
df_with_ebd.isnull().sum()

#%%

'''
EDA

F : ori 
'''

#%%
'''
temp = 0.1 , sim_score = (0.0, 0.1),  Ori-Gen1
'''
# 적으면
df_no_ebd[(df_no_ebd['F_0_1'] > 0) & (df_no_ebd['F_0_1'] < 0.1)]
df_no_ebd[(df_no_ebd['F_0_1'] > 0) & (df_no_ebd['F_0_1'] < 0.1)].to_csv("0.1-0~0.1-조회.csv", index=False, encoding='utf-8-sig')

# 많으면
idx = sampling(df_no_ebd, 'F_0_1', 0, 0.1, 30) # 0.1 and ori vs gen1
df_no_ebd.iloc[idx, :]
df_no_ebd.iloc[idx, :].to_csv("0.1-0~0.1-조회.csv", index=False, encoding='utf-8-sig')


# Ori
# 텍스트 평균 길이
df_no_ebd[(df_no_ebd['F_0_1'] > 0) & (df_no_ebd['F_0_1'] < 0.1)]['Text'].apply(len).mean()

# 텍스트 평균 토큰의 수
df_no_ebd[(df_no_ebd['F_0_1'] > 0) & (df_no_ebd['F_0_1'] < 0.1)]['Text'].apply(lambda x: len(x.split())).mean()


# Gen1
# 텍스트 평균 길이
df_no_ebd[(df_no_ebd['F_0_1'] > 0) & (df_no_ebd['F_0_1'] < 0.1)]['0.1-gen1'].apply(len).mean()

# 텍스트 평균 토큰의 수
df_no_ebd[(df_no_ebd['F_0_1'] > 0) & (df_no_ebd['F_0_1'] < 0.1)]['0.1-gen1'].apply(lambda x: len(x.split())).mean()

#%%
'''
temp = 0.1 , sim_score = (0.1, 0.2),  Ori-Gen1
'''

df_no_ebd[(df_no_ebd['F_0_1'] > 0.1) & (df_no_ebd['F_0_1'] < 0.2)]
df_no_ebd[(df_no_ebd['F_0_1'] > 0.1) & (df_no_ebd['F_0_1'] < 0.2)].to_csv("0.1-0.1~0.2-조회.csv", index=False, encoding='utf-8-sig')

# Ori
# 텍스트 평균 길이
df_no_ebd[(df_no_ebd['F_0_1'] > 0.1) & (df_no_ebd['F_0_1'] < 0.2)]['Text'].dropna().apply(len).mean()

# 텍스트 평균 토큰의 수
df_no_ebd[(df_no_ebd['F_0_1'] > 0.1) & (df_no_ebd['F_0_1'] < 0.2)]['Text'].dropna().apply(lambda x: len(x.split())).mean()

# Gen1
# 텍스트 평균 길이
df_no_ebd[(df_no_ebd['F_0_1'] > 0) & (df_no_ebd['F_0_1'] < 0.1)]['0.1-gen1'].dropna().apply(len).mean()

# 텍스트 평균 토큰의 수
df_no_ebd[(df_no_ebd['F_0_1'] > 0) & (df_no_ebd['F_0_1'] < 0.1)]['0.1-gen1'].dropna().apply(lambda x: len(x.split())).mean()

#%%

'''
temp = 0.1 , sim_score = (0.2, 0.3),  Ori-Gen1
'''

df_no_ebd[(df_no_ebd['F_0_1'] > 0.2) & (df_no_ebd['F_0_1'] < 0.3)].to_csv("0.1-0.2~0.3-조회.csv", index=False, encoding='utf-8-sig')


avg_length_ori, avg_token_count_ori = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['F_0_1'] > 0.2) & (df_no_ebd['F_0_1'] < 0.3)], 
                                                'F_0_1',
                                                0.2, 0.3,
                                                'Text')

avg_length_gen1, avg_token_count_gen1 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['F_0_1'] > 0.2) & (df_no_ebd['F_0_1'] < 0.3)], 
                                                'F_0_1',
                                                0.2, 0.3,
                                                '0.1-gen1')

print(f"Ori 텍스트 평균 길이: {avg_length_ori}")
print(f"Ori 텍스트 평균 토큰 수: {avg_token_count_ori}")

print(f"Gen1 텍스트 평균 길이: {avg_length_gen1}")
print(f"Gen1 텍스트 평균 토큰 수: {avg_token_count_gen1}")
#%%

'''
temp = 0.1 , sim_score = (0.4, 0.6),  Ori-Gen1
'''

df_no_ebd[(df_no_ebd['F_0_1'] >= 0.4) & (df_no_ebd['F_0_1'] < 0.6)]
df_no_ebd[(df_no_ebd['F_0_1'] >= 0.4) & (df_no_ebd['F_0_1'] < 0.6)].to_csv("0.1-0.4~0.6-조회.csv", index=False, encoding='utf-8-sig')


avg_length_ori, avg_token_count_ori = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['F_0_1'] >= 0.4) & (df_no_ebd['F_0_1'] < 0.6)], 
                                                'F_0_1',
                                                0.4, 0.6,
                                                'Text')

avg_length_gen1, avg_token_count_gen1 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['F_0_1'] >= 0.4) & (df_no_ebd['F_0_1'] < 0.6)], 
                                                'F_0_1',
                                                0.4, 0.6,
                                                '0.1-gen1')

print(avg_length_ori)
print(avg_token_count_ori)

print(avg_length_gen1)
print(avg_token_count_gen1)


#%%

'''
temp = 0.1 , sim_score = (0.6, 0.8),  Ori-Gen1
'''

df_no_ebd[(df_no_ebd['F_0_1'] >= 0.6) & (df_no_ebd['F_0_1'] < 0.8)]
df_no_ebd[(df_no_ebd['F_0_1'] >= 0.6) & (df_no_ebd['F_0_1'] < 0.8)].to_csv("0.1-0.6~0.8-조회.csv", index=False, encoding='utf-8-sig')


avg_length_ori, avg_token_count_ori = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['F_0_1'] >= 0.6) & (df_no_ebd['F_0_1'] < 0.8)], 
                                                'F_0_1',
                                                0.6, 0.8,
                                                'Text')

avg_length_gen1, avg_token_count_gen1 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['F_0_1'] >= 0.6) & (df_no_ebd['F_0_1'] < 0.8)], 
                                                'F_0_1',
                                                0.6, 0.8,
                                                '0.1-gen1')

print(avg_length_ori)
print(avg_token_count_ori)

print(avg_length_gen1)
print(avg_token_count_gen1)

#%%

df_no_ebd[(df_no_ebd['F_0_1'] >= 0.8)]
df_no_ebd[(df_no_ebd['F_0_1'] >= 0.8)].to_csv("0.1-0.8~-조회.csv", index=False, encoding='utf-8-sig')


avg_length_ori, avg_token_count_ori = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['F_0_1'] >= 0.8)], 
                                                'F_0_1',
                                                0.8, 1.0,
                                                'Text')

avg_length_gen1, avg_token_count_gen1 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['F_0_1'] >= 0.8)], 
                                                'F_0_1',
                                                0.8, 1.0,
                                                '0.1-gen1')

print(avg_length_ori)
print(avg_token_count_ori)

print(avg_length_gen1)
print(avg_token_count_gen1)

#%%
'''
temp = 0.1 , sim_score = (0., 0.3),  Gen1-Gen2
'''

df_no_ebd[(df_no_ebd['G_0_1'] >= 0) & (df_no_ebd['G_0_1'] < 0.3)]
df_no_ebd[(df_no_ebd['G_0_1'] >= 0) & (df_no_ebd['G_0_1'] < 0.3)].to_csv("0.1-0~0.3-조회.csv", index=False, encoding='utf-8-sig')


avg_length_gen1, avg_token_count_gen1 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['G_0_1'] >= 0) & (df_no_ebd['G_0_1'] < 0.3)], 
                                                'G_0_1',
                                                0, 0.3,
                                                '0.1-gen1')

avg_length_gen2, avg_token_count_gen2 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['G_0_1'] >= 0) & (df_no_ebd['G_0_1'] < 0.3)], 
                                                'G_0_1',
                                                0, 0.3,
                                                '0.1-gen2')


print(avg_length_gen1)
print(avg_token_count_gen1)

print(avg_length_gen2)
print(avg_token_count_gen2)

#%%

df_no_ebd[(df_no_ebd['G_0_1'] >= 0.3) & (df_no_ebd['G_0_1'] < 0.6)]
df_no_ebd[(df_no_ebd['G_0_1'] >= 0.3) & (df_no_ebd['G_0_1'] < 0.6)].to_csv("0.3-0~0.6-조회.csv", index=False, encoding='utf-8-sig')


avg_length_gen1, avg_token_count_gen1 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['G_0_1'] >= 0.3) & (df_no_ebd['G_0_1'] < 0.6)], 
                                                'G_0_1',
                                                0.3, 0.6,
                                                '0.1-gen1')

avg_length_gen2, avg_token_count_gen2 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['G_0_1'] >= 0.3) & (df_no_ebd['G_0_1'] < 0.6)], 
                                                'G_0_1',
                                                0.3, 0.6,
                                                '0.1-gen2')


print(avg_length_gen1)
print(avg_token_count_gen1)

print(avg_length_gen2)
print(avg_token_count_gen2)


# %%

# 0.6 ~ 0.9 구간으로 작업
df_no_ebd[(df_no_ebd['G_0_1'] >= 0.6) & (df_no_ebd['G_0_1'] < 0.9)]
df_no_ebd[(df_no_ebd['G_0_1'] >= 0.6) & (df_no_ebd['G_0_1'] < 0.9)].to_csv("0.1-0.6~0.9-조회.csv", index=False, encoding='utf-8-sig')

# Gen1 통계 계산
avg_length_gen1_06_09, avg_token_count_gen1_06_09 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['G_0_1'] >= 0.6) & (df_no_ebd['G_0_1'] < 0.9)], 
                                                'G_0_1',
                                                0.6, 0.9,
                                                '0.1-gen1')

# Gen2 통계 계산
avg_length_gen2_06_09, avg_token_count_gen2_06_09 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['G_0_1'] >= 0.6) & (df_no_ebd['G_0_1'] < 0.9)], 
                                                'G_0_1',
                                                0.6, 0.9,
                                                '0.1-gen2')

# 결과 출력
print(avg_length_gen1_06_09)
print(avg_token_count_gen1_06_09)

print(avg_length_gen2_06_09)
print(avg_token_count_gen2_06_09)
# %%

# 0.9 ~ 1.0 구간으로 작업
df_no_ebd[(df_no_ebd['G_0_1'] >= 0.9) & (df_no_ebd['G_0_1'] <= 1.0)]
df_no_ebd[(df_no_ebd['G_0_1'] >= 0.9) & (df_no_ebd['G_0_1'] <= 1.0)].to_csv("0.1-0.9~1.0-조회.csv", index=False, encoding='utf-8-sig')

# Gen1 통계 계산
avg_length_gen1_09_10, avg_token_count_gen1_09_10 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['G_0_1'] >= 0.9) & (df_no_ebd['G_0_1'] <= 1.0)], 
                                                'G_0_1',
                                                0.9, 1.0,
                                                '0.1-gen1')

# Gen2 통계 계산
avg_length_gen2_09_10, avg_token_count_gen2_09_10 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['G_0_1'] >= 0.9) & (df_no_ebd['G_0_1'] <= 1.0)], 
                                                'G_0_1',
                                                0.9, 1.0,
                                                '0.1-gen2')

# 결과 출력
print(avg_length_gen1_09_10)
print(avg_token_count_gen1_09_10)

print(avg_length_gen2_09_10)
print(avg_token_count_gen2_09_10)
# %%
'''
Temp = 0.7 해보기, H에 관해서 하자.
'''

# 0.0 ~ 0.3 구간으로 작업 (H_0_7 컬럼 기준)
df_no_ebd[(df_no_ebd['H_0_7'] >= 0.0) & (df_no_ebd['H_0_7'] < 0.3)]
df_no_ebd[(df_no_ebd['H_0_7'] >= 0.0) & (df_no_ebd['H_0_7'] < 0.3)].to_csv("0.1-0~0.3-조회.csv", index=False, encoding='utf-8-sig')

# Gen1 통계 계산
avg_length_gen1_00_03, avg_token_count_gen1_00_03 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['H_0_7'] >= 0.0) & (df_no_ebd['H_0_7'] < 0.3)], 
                                                'H_0_7',
                                                0.0, 0.3,
                                                '0.7-gen1')

# Gen2 통계 계산
avg_length_gen2_00_03, avg_token_count_gen2_00_03 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['H_0_7'] >= 0.0) & (df_no_ebd['H_0_7'] < 0.3)], 
                                                'H_0_7',
                                                0.0, 0.3,
                                                '0.7-gen1-1')

# 결과 출력
print(avg_length_gen1_00_03)
print(avg_token_count_gen1_00_03)

print(avg_length_gen2_00_03)
print(avg_token_count_gen2_00_03)

#%%

# 0.3 ~ 0.6 구간으로 작업 (H_0_7 컬럼 기준)
df_no_ebd[(df_no_ebd['H_0_7'] >= 0.3) & (df_no_ebd['H_0_7'] < 0.6)]
df_no_ebd[(df_no_ebd['H_0_7'] >= 0.3) & (df_no_ebd['H_0_7'] < 0.6)].to_csv("0.7-0.3~0.6-조회.csv", index=False, encoding='utf-8-sig')

# Gen1 통계 계산
avg_length_gen1_03_06, avg_token_count_gen1_03_06 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['H_0_7'] >= 0.3) & (df_no_ebd['H_0_7'] < 0.6)], 
                                                'H_0_7',
                                                0.3, 0.6,
                                                '0.7-gen1')

# Gen2 통계 계산
avg_length_gen2_03_06, avg_token_count_gen2_03_06 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['H_0_7'] >= 0.3) & (df_no_ebd['H_0_7'] < 0.6)], 
                                                'H_0_7',
                                                0.3, 0.6,
                                                '0.7-gen1-1')

# 결과 출력
print(avg_length_gen1_03_06)
print(avg_token_count_gen1_03_06)

print(avg_length_gen2_03_06)
print(avg_token_count_gen2_03_06)
# %%

# 0.9 ~ 1.0 구간으로 작업 (H_0_7 컬럼 기준)
df_no_ebd[(df_no_ebd['H_0_7'] >= 0.9) & (df_no_ebd['H_0_7'] <= 1.0)]
df_no_ebd[(df_no_ebd['H_0_7'] >= 0.9) & (df_no_ebd['H_0_7'] <= 1.0)].to_csv("0.7-0.9~1.0-조회.csv", index=False, encoding='utf-8-sig')

# Gen1 통계 계산
avg_length_gen1_09_10, avg_token_count_gen1_09_10 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['H_0_7'] >= 0.9) & (df_no_ebd['H_0_7'] <= 1.0)], 
                                                'H_0_7',
                                                0.9, 1.0,
                                                '0.7-gen1')

# Gen2 통계 계산
avg_length_gen2_09_10, avg_token_count_gen2_09_10 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['H_0_7'] >= 0.9) & (df_no_ebd['H_0_7'] <= 1.0)], 
                                                'H_0_7',
                                                0.9, 1.0,
                                                '0.7-gen1-1')

# 결과 출력
print(avg_length_gen1_09_10)
print(avg_token_count_gen1_09_10)

print(avg_length_gen2_09_10)
print(avg_token_count_gen2_09_10)

#%%

# 0.0 ~ 0.3 구간으로 작업 (H_1_5 컬럼 기준)
df_no_ebd[(df_no_ebd['H_1_5'] >= 0.0) & (df_no_ebd['H_1_5'] < 0.3)]
df_no_ebd[(df_no_ebd['H_1_5'] >= 0.0) & (df_no_ebd['H_1_5'] < 0.3)].to_csv("1.5-0~0.3-조회.csv", index=False, encoding='utf-8-sig')

# Gen1 통계 계산
avg_length_gen1_00_03, avg_token_count_gen1_00_03 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['H_1_5'] >= 0.0) & (df_no_ebd['H_1_5'] < 0.3)], 
                                                'H_1_5',
                                                0.0, 0.3,
                                                '1.5-gen1')

# Gen2 통계 계산
avg_length_gen2_00_03, avg_token_count_gen2_00_03 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['H_1_5'] >= 0.0) & (df_no_ebd['H_1_5'] < 0.3)], 
                                                'H_1_5',
                                                0.0, 0.3,
                                                '1.5-gen1-1')

# 결과 출력
print(avg_length_gen1_00_03)
print(avg_token_count_gen1_00_03)

print(avg_length_gen2_00_03)
print(avg_token_count_gen2_00_03)


#%%

# 0.3 ~ 0.6 구간으로 작업 (H_1_5 컬럼 기준)
df_no_ebd[(df_no_ebd['H_1_5'] >= 0.3) & (df_no_ebd['H_1_5'] < 0.6)]
df_no_ebd[(df_no_ebd['H_1_5'] >= 0.3) & (df_no_ebd['H_1_5'] < 0.6)].to_csv("1.5-0.3~0.6-조회.csv", index=False, encoding='utf-8-sig')

# Gen1 통계 계산
avg_length_gen1_03_06, avg_token_count_gen1_03_06 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['H_1_5'] >= 0.3) & (df_no_ebd['H_1_5'] < 0.6)], 
                                                'H_1_5',
                                                0.3, 0.6,
                                                '1.5-gen1')

# Gen2 통계 계산
avg_length_gen2_03_06, avg_token_count_gen2_03_06 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['H_1_5'] >= 0.3) & (df_no_ebd['H_1_5'] < 0.6)], 
                                                'H_1_5',
                                                0.3, 0.6,
                                                '1.5-gen1-1')

# 결과 출력
print(avg_length_gen1_03_06)
print(avg_token_count_gen1_03_06)

print(avg_length_gen2_03_06)
print(avg_token_count_gen2_03_06)
#%%

# 0.6 ~ 1.0 구간으로 작업 (H_1_5 컬럼 기준)
df_no_ebd[(df_no_ebd['H_1_5'] >= 0.6) & (df_no_ebd['H_1_5'] <= 1.0)]
df_no_ebd[(df_no_ebd['H_1_5'] >= 0.6) & (df_no_ebd['H_1_5'] <= 1.0)].to_csv("1.5-0.6~1.0-조회.csv", index=False, encoding='utf-8-sig')

# Gen1 통계 계산
avg_length_gen1_06_10, avg_token_count_gen1_06_10 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['H_1_5'] >= 0.6) & (df_no_ebd['H_1_5'] <= 1.0)], 
                                                'H_1_5',
                                                0.6, 1.0,
                                                '1.5-gen1')

# Gen2 통계 계산
avg_length_gen2_06_10, avg_token_count_gen2_06_10 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['H_1_5'] >= 0.6) & (df_no_ebd['H_1_5'] <= 1.0)], 
                                                'H_1_5',
                                                0.6, 1.0,
                                                '1.5-gen1-1')

# 결과 출력
print(avg_length_gen1_06_10)
print(avg_token_count_gen1_06_10)

print(avg_length_gen2_06_10)
print(avg_token_count_gen2_06_10)







#%%

# 0.9 ~ 1.0 구간으로 작업 (H_1_5 컬럼 기준)
df_no_ebd[(df_no_ebd['H_1_5'] >= 0.9) & (df_no_ebd['H_1_5'] <= 1.0)]
df_no_ebd[(df_no_ebd['H_1_5'] >= 0.9) & (df_no_ebd['H_1_5'] <= 1.0)].to_csv("1.5-0.9~1.0-조회.csv", index=False, encoding='utf-8-sig')

# Gen1 통계 계산
avg_length_gen1_09_10, avg_token_count_gen1_09_10 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['H_1_5'] >= 0.9) & (df_no_ebd['H_1_5'] <= 1.0)], 
                                                'H_1_5',
                                                0.9, 1.0,
                                                '1.5-gen1')

# Gen2 통계 계산
avg_length_gen2_09_10, avg_token_count_gen2_09_10 = cal_text_stats(
                                                df_no_ebd[(df_no_ebd['H_1_5'] >= 0.9) & (df_no_ebd['H_1_5'] <= 1.0)], 
                                                'H_1_5',
                                                0.9, 1.0,
                                                '1.5-gen1-1')

# 결과 출력
print(avg_length_gen1_09_10)
print(avg_token_count_gen1_09_10)

print(avg_length_gen2_09_10)
print(avg_token_count_gen2_09_10)
