'''
실제 텍스트 임베딩과 생성된 텍스트 임베딩과의 유사도 비교
'''


#%%

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv



#%%
'''
1단계 생성 텍스트 임베딩
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

def get_embedding(text, model="text-embedding-3-small"):
    if isinstance(text, str):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding
    else:
        return np.zeros(1536)  # 빈 텍스트나 NaN 값에 대한 기본 임베딩 벡터 (예: 1536차원 벡터)
    

os.chdir('C:/Users/mose/agoda/data/')

df = pd.read_csv("데이터이름을 입력하시오.csv")

# tqdm과 pandas 통합
tqdm.pandas()

# 진행 상황 모니터링을 위한 코드 수정
df["gen-embd"] = df["Text"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))

# 최종 결과 저장
df.to_csv("24-08-07-생성텍스트임베딩추가.csv", index=False, encoding='utf-8-sig')


#%%

'''
2단계 유사도 점수 계산
'''

#%%

# 문자열 데이터를 리스트로 변환하고 각 원소를 float으로 변환하는 함수 정의
def convert_to_float_list(string_data):
    try:
        # 문자열을 리스트로 변환
        data_list = json.loads(string_data)
        # 리스트의 각 원소를 float으로 변환
        float_list = [float(i) for i in data_list]
        return float_list
    except json.JSONDecodeError:
        return []
    
# 데이터를 청크로 나누어 코사인 유사도를 계산하는 함수 정의
def calculate_cosine_similarity_in_chunks(embd1, embd2, chunk_size=1000):
    cosine_similarities = []
    for start in range(0, len(embd1), chunk_size):
        end = min(start + chunk_size, len(embd1))
        chunk_cosine_similarities = cosine_similarity(embd1[start:end], embd2[start:end])
        cosine_similarities.extend(chunk_cosine_similarities.diagonal())
    return cosine_similarities


os.chdir('C:/Users/UOS/proj_0/Agoda/GPT-Application/')

df1 = pd.read_csv("24-07-29-임베딩추가.csv", encoding='utf-8-sig')
df2 = pd.read_csv("24-07-31-r-임베딩추가.csv", encoding='utf-8-sig')


len(df1)
len(df2)


df1.isnull().sum()
df2.isnull().sum()

# 특정칼럼명 변경
df1.rename(columns={'embedding': 'embd'}, inplace=True)


df1.columns
df2.columns

df1 = df1.dropna(subset=['Text'])

len(df1), len(df2)

# 데이터프레임의 각 행에 대해 변환 함수 적용
df1['embd'] = df1['embd'].apply(convert_to_float_list)
df2['r-embd'] = df2['r-embd'].apply(convert_to_float_list)

embd1 = df1['embd'].to_list()
embd2 = df2['r-embd'].to_list()

# 코사인 유사도 계산
cosine_similarities = calculate_cosine_similarity_in_chunks(embd1, embd2, chunk_size=1000)

len(cosine_similarities)

df2['cosine_similarity'] = cosine_similarities
