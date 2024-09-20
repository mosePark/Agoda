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
os.chdir('C:/Users/UOS/Desktop/Agoda-Data/') # lab
# os.chdir('/home1/mose1103/agoda/LLM') # UBAI

df = pd.read_csv("all-0.1-gen1_final.csv", index_col='Unnamed: 0')

#%%
'''
Prompt 1-1 - 생성1 리뷰를 기반으로 유사한 문맥의 리뷰를 생성
'''

prompt_template = """
Read the following review text and generate another review text with a similar context:

Note: The original text is composed in the form of title + "\n" + body, as shown below:
Original Review: "{GeneratedReview}"

Please generate a review in a similar format, with a title and a body:

Generated Review:
"""

# 입력 변수를 사용하여 프롬프트 템플릿 생성
template = PromptTemplate(template=prompt_template, input_variables=["GeneratedReview"])

# OpenAI 모델 설정
llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo", temperature=0.1)
llm_chain = LLMChain(llm=llm, prompt=template)


new_generated_reviews = []

# 원본 리뷰 데이터를 기반으로 비슷한 문맥의 리뷰 생성 (gen1)
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Reviews"):
    review = row['generated_review']  # 'Text' 열 참조
    result = llm_chain.run({"GeneratedReview": review})
    new_generated_review = result.strip()
    new_generated_reviews.append(new_generated_review)
    
    # 생성된 리뷰를 데이터프레임의 해당 행에 추가
    df.at[index, 'new_generated_review'] = new_generated_review
    
    # 현재 생성된 리뷰를 파일로 저장
    df.to_csv("all-0.1-gen1-1.csv", encoding='utf-8-sig')

# 최종 생성된 리뷰를 파일로 안전하게 저장
df.to_csv("all-0.1-gen1-1_final.csv", encoding='utf-8-sig')

# 아이클라우드에도 저장

os.chdir('c:/Users/UOS/iCloudDrive')
df.to_csv("all-0.1-gen1-1_final.csv", encoding='utf-8-sig')
