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
# os.chdir('C:/Users/mose/agoda/data/') # home
os.chdir('C:/Users/UOS/Desktop/Agoda-Data/') # lab
# os.chdir('/home1/mose1103/agoda/LLM') # UBAI

df = pd.read_csv("new_gen_reviews_final.csv", index_col='Unnamed: 0')

df.isnull().sum()

'''
임베딩벡터 생성
'''
# tqdm과 pandas 통합
tqdm.pandas()

# 진행 상황 모니터링을 위한 코드 수정
df["gen1-2"] = df["new_generated_review"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))

# 최종 결과 저장
df.to_csv("gen1-2.csv", index=False, encoding='utf-8-sig')
