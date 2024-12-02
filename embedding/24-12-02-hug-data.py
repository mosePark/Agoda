#%%
import pandas as pd
from datasets import load_dataset


#%%
dataset = load_dataset("humarin/chatgpt-paraphrases")

df = dataset['train'].to_pandas()


source_counts = df['source'].value_counts()


# 'source'별 10% 샘플링
sampled_df = df.groupby('source').apply(lambda x: x.sample(frac=0.1, random_state=42)).reset_index(drop=True)

# 결과 확인
print("샘플링된 데이터셋 크기:")
print(sampled_df['source'].value_counts())
