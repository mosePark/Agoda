'''
all gen text 에 대해 임베딩 작업
temp parameter : {0.1, 0.7, 1.5}
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

os.chdir('C:/Users/mose/agoda/data/all') # home
# os.chdir('C:/Users/UOS/Desktop/Agoda-Data/') # lab
# os.chdir('/home1/mose1103/agoda/LLM') # UBAI


df_1 = pd.read_csv("all-0.1-gen1_.csv", index_col='Unnamed: 0')
df_2 = pd.read_csv("all-0.1-gen2_.csv", index_col='Unnamed: 0')
df_3 = pd.read_csv("all-0.1-gen1-1_.csv", index_col='Unnamed: 0')
df_4 = pd.read_csv("all-0.7-gen1_.csv", index_col='Unnamed: 0')
df_5 = pd.read_csv("all-0.7-gen2_.csv", index_col='Unnamed: 0')
df_6 = pd.read_csv("all-0.7-gen1-1_.csv", index_col='Unnamed: 0')
df_7 = pd.read_csv("all-1.5-gen1_.csv", index_col='Unnamed: 0')
df_8 = pd.read_csv("all-1.5-gen2_.csv", index_col='Unnamed: 0')
# df_9 = pd.read_csv("all-1.5-gen1-1_.csv", index_col='Unnamed: 0') # 아직 진행되지 못함

dfs = {
    'df_1': df_1,
    'df_2': df_2,
    'df_3': df_3,
    'df_4': df_4,
    'df_5': df_5,
    'df_6': df_6,
    'df_7': df_7,
    'df_8': df_8,
    # 'df_9': df_9
}


for name, df in dfs.items():
    print(f"{name} columns: {df.columns.tolist()}")


'''
임베딩
'''

# tqdm과 pandas 통합
tqdm.pandas()

os.chdir('C:/Users/mose/agoda/data/all/all-ebd') # 저장링크

# 임베딩 벡터 생성 및 새 변수 추가, 파일 저장
df_1["0.1-gen1-ebd"] = df_1["generated_review"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 0.1 gen1-1
df_1.to_csv("0.1-gen1-ebd.csv", index=True, encoding='utf-8-sig')

df_2["0.1-gen2-ebd"] = df_2["generated_review_2"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 0.1 gen2
df_2.to_csv("0.1-gen2-ebd.csv", index=True, encoding='utf-8-sig')

df_3["0.1-gen1-1-ebd"] = df_3["new_generated_review"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 1.5 gen1
df_3.to_csv("0.1-gen1-1-ebd.csv", index=True, encoding='utf-8-sig')

df_4["0.7-gen1-ebd"] = df_4["generated_review"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 1.5 gen1-1
df_4.to_csv("0.7-gen1-ebd.csv", index=True, encoding='utf-8-sig')

df_5["0.7-gen2-ebd"] = df_5["generated_review_2"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 1.5 gen2
df_5.to_csv("0.7-gen2-ebd.csv", index=True, encoding='utf-8-sig')

df_6["0.7-gen1-1-ebd"] = df_1["new_generated_review"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 0.1 gen1-1
df_6.to_csv("0.7-gen1-1-ebd.csv", index=True, encoding='utf-8-sig')

df_7["1.5-gen1-ebd"] = df_2["generated_review"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 0.1 gen2
df_7.to_csv("1.5-gen1-ebd.csv", index=True, encoding='utf-8-sig')

df_8["1.5-gen2-ebd"] = df_3["generated_review_2"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 1.5 gen1
df_8.to_csv("1.5-gen2-ebd.csv", index=True, encoding='utf-8-sig')

# df_9["1.5-gen1-1-ebd"] = df_4["new_generated_review"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))  # 1.5 gen1-1
# df_9.to_csv("1.5-gen1-1-ebd.csv", index=True, encoding='utf-8-sig')
