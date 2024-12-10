#%%

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from langchain import OpenAI, LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

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

#%%
'''
데이터 로드
'''

# os.chdir('C:/Users/mose/agoda/data/') # home
# os.chdir('C:/Users/UOS/Desktop/Agoda-Data/') # lab
os.chdir('/home1/mose1103/agoda/LLM') # UBAI
# os.chdir('C:/Users/UOS/Desktop/')


df = pd.read_csv("all-0.1-hug-gen1.csv", encoding='utf-8-sig', index_col='Unnamed: 0')

df.rename(columns={'0.1-gen1': 'generated_review'}, inplace=True)

#%%

'''
Prompt - 원본 텍스트를 기반으로 유사한 문맥의 텍스트를 생성
'''
prompt_template = """
Read the following text and generate another text with a similar context:

Note: The original text is composed of a single sentence in the form of either a question or a statement. as shown below:
Original text: "{GeneratedReview}"

Please generate a text in a similar format:

Generated text:
"""

# 입력 변수를 사용하여 프롬프트 템플릿 생성
template = PromptTemplate(template=prompt_template, input_variables=["GeneratedReview"])

# 주어진 API 키와 모델 이름으로 OpenAI LLM 초기화
llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo", temperature=0.1)
llm_chain = LLMChain(llm=llm, prompt=template)

new_generated_reviews = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Reviews"):
    review = row['generated_review']
    result = llm_chain.run({"GeneratedReview": review})
    new_generated_review = result.strip()
    new_generated_reviews.append(new_generated_review)
    
    df.at[index, 'new_generated_review'] = new_generated_review

    
    # 현재 생성된 리뷰를 파일로 저장
    df.to_csv("0.1-hug-gen1-1.csv", encoding='utf-8-sig')

# 최종 생성된 리뷰를 파일로 안전하게 저장
df.to_csv("all-0.1-hug-gen1-1.csv", encoding='utf-8-sig')
