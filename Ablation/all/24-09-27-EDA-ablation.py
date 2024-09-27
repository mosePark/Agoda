'''
유사도값 별로 어떤 데이터인지 살펴보기
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sampling(dataframe, column_name, lower_bound, upper_bound, sample_size):

    filtered_data = dataframe[(dataframe[column_name] > lower_bound) & (dataframe[column_name] < upper_bound)]
    
    if len(filtered_data) <= sample_size:
        random_indices = filtered_data.index
    else:
        random_indices = np.random.choice(filtered_data.index, size=sample_size, replace=False)
    
    return random_indices


#%%
'''
데이터 로드
'''
# os.chdir('C:/Users/mose/agoda/data/') # home
# os.chdir('C:/Users/UOS/Desktop/Agoda-Data') # lab
# os.chdir('/home1/mose1103/agoda/LLM') # UBAI
os.chdir('E:/mose/data/ablation2') # D-drive

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

F_full = pd.read_csv("F_full.csv", encoding='utf-8-sig')
G_full = pd.read_csv("G_full.csv", encoding='utf-8-sig')
H_full = pd.read_csv("H_full.csv", encoding='utf-8-sig')

#%%

idx = sampling(F_full, 'F_full', 0, 0.1, 30) # 0.1 and ori vs gen1

df_1.iloc[idx]

df_1['Full'].iloc[idx]
df_1['generated_review'].iloc[idx]

df_1.iloc[idx].to_excel("0.1-F-text-조회.xlsx")


df_1[df_1['Text'].str.len() < 4].to_excel("텍스트길이가 4미만조회.xlsx")
