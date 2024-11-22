#%%
import os
import numpy as np
import pandas as pd



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
