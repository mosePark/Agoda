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
