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

os.getcwd()

sdf = pd.read_csv('hug.csv', encoding='utf-8-sig')

#%%

'''
API 로드
'''

# API KEY 정보로드
current_directory = os.getcwd()
project_root = current_directory

# .env 파일 경로 설정
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)

# API 키를 환경 변수에서 가져오기
api_key = os.getenv("OPENAI_API_KEY")

'''
임베딩 함수 정의
'''

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    if isinstance(text, str):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding
    else:
        return np.zeros(1536)




#%%

tqdm.pandas()

sdf["ori-ebd"] = sdf["text"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # ori
sdf["0.7-gen1-ebd"] = sdf["gen1"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 0.7 gen1
sdf["0.7-gen2-ebd"] = sdf["gen2"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 0.7 gen2

sdf.to_csv('hug-end.csv', encoding='utf-8-sig', index=False)
