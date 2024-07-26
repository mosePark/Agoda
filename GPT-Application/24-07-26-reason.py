'''
Prompt : 
'''

import os

import numpy as np
import pandas as pd

from tqdm import tqdm

from langchain import OpenAI, LLMChain
from langchain import PromptTemplate, LLMChain
from langchain_openai import ChatOpenAI
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
데이터 로드
'''
os.chdir('C:/Users/UOS/proj_0/Agoda/GPT-Application/')

df = pd.read_csv("agoda.csv")


'''
Prompt
'''

# 점수 예측과 이유를 위한 프롬프트 템플릿을 정의합니다.
prompt_template = """
Predict the score (out of 10) for the following review text and provide a reason for the score in two sentences or less:

Review: "{review}"

Score (1-10):
Reason:
"""

# 템플릿을 사용하여 프롬프트 생성
template = PromptTemplate(template=prompt_template, input_variables=["review"])

# OpenAI LLM과 프롬프트 템플릿을 사용하여 LLMChain을 만듭니다.
llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo")
llm_chain = LLMChain(llm=llm, prompt=template)

# 예측 결과를 저장할 리스트 초기화
predicted_scores = []
predicted_reasons = []

# 리뷰 데이터 정보를 입력하고 점수와 이유를 예측합니다.
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Reviews"):
    review = row['Text']
    result = llm_chain.run({"review": review})
    predicted_score = result.split("Reason:")[0].strip()
    predicted_reason = result.split("Reason:")[1].strip()
    predicted_scores.append(predicted_score)
    predicted_reasons.append(predicted_reason)
    
    # 예측 결과를 데이터프레임의 해당 행에 추가
    df.at[index, 'y_hat'] = predicted_score
    df.at[index, 'Reason'] = predicted_reason
    
    # 현재까지의 예측 결과를 파일로 저장
    df.to_csv("Review+Reason.csv", index=False, encoding='utf-8-sig')

# 최종 예측 결과를 파일로 저장 (안전하게)
df.to_csv("Review+Reason_Final.csv", index=False, encoding='utf-8-sig')
