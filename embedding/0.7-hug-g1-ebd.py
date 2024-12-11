#%%

import os
import ast
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv


def get_embedding(text, model="text-embedding-3-small"):
    if isinstance(text, str):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding
    else:
        return np.zeros(1536)

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

client = OpenAI()


#%%

df = pd.read_csv("all-0.4-hug-gen1.csv", encoding='utf-8-sig', index_col='Unnamed: 0')


#%%


tqdm.pandas()

df["0.4-gen1-ebd"] = df["generated_review"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))


df.to_csv("0.4-hug-gen1-ebd.csv", encoding='utf-8-sig')
