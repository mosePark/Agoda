'''
temp parameter : 0.1, 1.5 임베딩 작업
'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

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
        return np.zeros(1536)  # 빈 텍스트나 NaN 값에 대한 기본 임베딩 벡터 (예: 1536차원 벡터)

'''
데이터 로드
'''
os.chdir('C:/Users/mose/agoda/data/') # home
# os.chdir('C:/Users/UOS/Desktop/Agoda-Data/') # lab
# os.chdir('/home1/mose1103/agoda/LLM') # UBAI

df_1 = pd.read_csv("0.1-gen1-1.csv", index_col='Unnamed: 0')
df_2 = pd.read_csv("0.1-gen2.csv", index_col='Unnamed: 0')
df_3 = pd.read_csv("1.5-gen1.csv", index_col='Unnamed: 0')
df_4 = pd.read_csv("1.5-gen1-1.csv", index_col='Unnamed: 0')
df_5 = pd.read_csv("1.5-gen2.csv", index_col='Unnamed: 0')

# df_1, df_2, df_3, df_4, df_5

df_1['new_generated_review'] # 0.1 gen1-1
df_2['generated_review_2'] # 0.1 gen2
df_3['generated_review'] # 1.5 gen1
df_4['new_generated_review'] # 1.5 gne1-1
df_5['generated_review_2'] # 1.5 gen2

'''
임베딩
'''

# tqdm과 pandas 통합
tqdm.pandas()

# 임베딩 벡터 생성 및 새 변수 추가
df_1["0.1-gen1-1-ebd"] = df_1["new_generated_review"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 0.1 gen1-1
df_2["0.1-gen2-ebd"] = df_2["generated_review_2"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 0.1 gen2
df_3["1.5-gen1-ebd"] = df_3["generated_review"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 1.5 gen1
df_4["1.5-gen1-1-ebd"] = df_4["new_generated_review"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 1.5 gen1-1
df_5["1.5-gen2-ebd"] = df_5["generated_review_2"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 1.5 gen2

# 결과 CSV 파일로 저장

os.chdir('C:/Users/mose/agoda/data/ebd/') # 저장링크

df_1.to_csv("0.1-gen1-1-ebd.csv", index=True, encoding='utf-8-sig')
df_2.to_csv("0.1-gen2-ebd.csv", index=True, encoding='utf-8-sig')
df_3.to_csv("1.5-gen1-ebd.csv", index=True, encoding='utf-8-sig')
df_4.to_csv("1.5-gen1-1-ebd.csv", index=True, encoding='utf-8-sig')
df_5.to_csv("1.5-gen2-ebd.csv", index=True, encoding='utf-8-sig')
