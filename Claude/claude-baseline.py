#%% 
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from langchain_anthropic import ChatAnthropic
from langchain import LLMChain
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

# Claude API 키를 환경 변수에서 가져오기
claude_api_key = os.getenv("CLAUDE_API_KEY")

#%%
'''
데이터 로드
'''

# os.chdir('C:/Users/mose/agoda/data/') # home
os.chdir('C:/Users/UOS/Desktop/Agoda-Data/raw') # lab

df = pd.read_csv("agoda2.csv")

os.chdir('C:/Users/UOS/Desktop/Agoda-Data/')
#%%

'''
Prompt
'''
# 비슷한 문맥의 리뷰를 생성하기 위한 프롬프트 템플릿 정의
prompt_template = """
Read the following review text and generate another review text with a similar context:

Note: The original text is composed in the form of title + "\n" + body, as shown below:
Original Review: "{Full}"

Please generate a review in a similar format, with a title and a body:

Generated Review:
"""

# 입력 변수를 사용하여 프롬프트 템플릿 생성
template = PromptTemplate(template=prompt_template, input_variables=["Full"])

# 주어진 API 키와 모델 이름으로 Claude LLM 초기화
llm = ChatAnthropic(api_key=claude_api_key, model_name="claude-2")
llm_chain = LLMChain(llm=llm, prompt=template)

# 생성된 리뷰를 저장할 리스트 초기화
generated_reviews = []

# 원본 리뷰 데이터를 기반으로 비슷한 문맥의 리뷰 생성
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Reviews"):
    review = row['Full']  # 'Full' 열 참조
    result = llm_chain.run({"Full": review})
    generated_review = result.strip()
    generated_reviews.append(generated_review)
    
    # 생성된 리뷰를 데이터프레임의 해당 행에 추가
    df.at[index, 'generated_review'] = generated_review
    
    # 현재 생성된 리뷰를 파일로 저장
    df.to_csv("claude_gen1.csv", encoding='utf-8-sig')

# 최종 생성된 리뷰를 파일로 안전하게 저장
df.to_csv("claude_gen1_final.csv", encoding='utf-8-sig')
