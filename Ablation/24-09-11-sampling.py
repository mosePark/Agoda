import os
import numpy as np
import pandas as pd

'''
데이터 로드
'''

# os.chdir('C:/Users/mose/agoda/data/') # home
os.chdir('C:/Users/UOS/Desktop/Agoda-Data/raw') # lab
# os.chdir('/home1/mose1103/agoda/LLM') # UBAI

df = pd.read_csv("agoda2.csv")

df = df.sample(n=3000, random_state=240911)

df.isnull().sum()

df.to_csv("sample.csv", index=True, encoding='utf-8-sig')


#%%

temp_0 = pd.read_csv('temp0.1_final.csv', index_col='Unnamed: 0')


temp_0_gen1 = temp_0.loc[df.index]

temp_0_gen1.to_csv("0.1-gen1.csv", index=True, encoding='utf-8-sig')

#%%

os.chdir('C:/Users/UOS/Desktop/Agoda-Data/')

temp_7 = pd.read_csv('gen1-gen2-ebd.csv', index_col='Unnamed: 0')
temp_7_1 = pd.read_csv('gen1-1-ebd.csv')


temp = pd.merge(temp_7, temp_7_1, how='outer', on=[col for col in temp_7.columns if col in temp_7_1.columns], suffixes=('', '_duplicate'))

abc = temp.loc[df.index]

abc.to_csv("0.7-all.csv", index=True, encoding='utf-8-sig')
