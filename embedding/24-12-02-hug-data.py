#%%

from datasets import load_dataset

import os
import ast
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv


#%%

'''
데이터 로드
'''

dataset = load_dataset("humarin/chatgpt-paraphrases")

df = dataset['train'].to_pandas()

df['source'].value_counts()
df['category'].value_counts()



# 10% 샘플링
sdf = (
    df.groupby(['source', 'category'], group_keys=False)
      .apply(lambda x: x.sample(frac=0.1, random_state=413))
)

idx = sdf.index

sdf['source'].value_counts()
sdf['category'].value_counts()

sdf["paraphrases"] = sdf["paraphrases"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
print(sdf["paraphrases"].head())

sdf["gen1"] = sdf["paraphrases"].apply(lambda x: x[0] if len(x) > 0 else None)
sdf["gen2"] = sdf["paraphrases"].apply(lambda x: x[1] if len(x) > 1 else None)

sdf.columns

sdf.info()
sdf.head()

sdf.iloc[1, :]['paraphrases']
sdf.iloc[1, :]['gen1']
sdf.iloc[1, :]['gen2']


# sdf_with_idx = sdf.reset_index().rename(columns={'index': 'idx'})
# sdf_with_idx.to_csv('hug.csv', encoding='utf-8-sig', index=False)
# sdf_with_idx.to_excel('hug.xlsx', index=False)

#%%

ori = pd.read_csv("hug-ori.csv", encoding='utf-8-sig')
g1 = pd.read_csv("hug-g1.csv", encoding='utf-8-sig')
g2 = pd.read_csv("hug-g2.csv", encoding='utf-8-sig')

ori.isnull().sum()
g1.isnull().sum()
g2.isnull().sum()

df = ori

df['0.7-gen1-ebd'] = g1['0.7-gen1-ebd']
df['0.7-gen2-ebd'] = g2['0.7-gen2-ebd']

# df.to_csv("hug-ebd.csv", encoding='utf-8-sig', index=False)

df.isnull().sum()

df = pd.read_csv("hug-ebd.csv", encoding='utf-8-sig')

df.columns

df['source'].value_counts()

quora = df[df['source'] == 'quora'] # 24714
squad2 = df[df['source'] == 'squad_2'] # 9198
cnn = df[df['source'] == 'cnn_news'] # 8008

quora.to_csv("quora-ebd.csv", encoding='utf-8-sig', index=False)
squad2.to_csv("squad2-ebd.csv", encoding='utf-8-sig', index=False)
cnn.to_csv("cnn-ebd.csv", encoding='utf-8-sig', index=False)
