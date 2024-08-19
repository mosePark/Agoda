'''
각 full text에서 각 카테고리에 해당하는 키워드를 추출해 리스트로 저장
'''
#%%

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from langchain import OpenAI, LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import ast

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

os.chdir('C:/Users/UOS/proj_0/Agoda/GPT-Application/')

df = pd.read_csv("agoda2.csv")


#%%

'''
Prompt
'''
# 프롬프트 템플릿 정의
prompt_template = """
Read the following review full text and extract keywords related to the following categories:
- Loc: Keywords related to location.
- Clean: Keywords related to cleanliness.
- Serv: Keywords related to service.
- Fac: Keywords related to facilities.
- VfM: Keywords related to value for money.

Only return keywords for each category as a list. If there are no relevant keywords, return an empty list.

Full Text: "{full_text}"

Extracted Keywords:
- Loc: 
- Clean:
- Serv:
- Fac:
- VfM:
"""

# 입력 변수를 사용하여 프롬프트 템플릿 생성
template = PromptTemplate(template=prompt_template, input_variables=["full_text"])

# 주어진 API 키와 모델 이름으로 OpenAI LLM 초기화
llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo")
llm_chain = LLMChain(llm=llm, prompt=template)

# LLM 결과를 저장할 리스트 초기화
results = []

# 원본 리뷰 데이터를 기반으로 GPT 출력 저장
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Reviews"):
    full_text = row['Full']  # 'Full' 열 참조
    result = llm_chain.run({"full_text": full_text})
    
    results.append(result)

# GPT 출력 결과를 새로운 열로 추가
df.loc[:, 'LLM_result'] = results

# 작업 중간 결과를 파일로 저장
df.to_csv("keyword_extraction.csv", index=False, encoding='utf-8-sig')

# 최종 결과를 파일로 저장
df.to_csv("keyword_extraction_final.csv", index=False, encoding='utf-8-sig')



#%% 연습용

# # 테스트를 위해 첫 20개의 행만 사용
# df_test = df.head(20).copy()  # 슬라이스한 데이터프레임을 명시적으로 복사

# # 프롬프트 템플릿 정의
# prompt_template = """
# Read the following review full text and extract keywords related to the following categories:
# - Loc: Keywords related to location.
# - Clean: Keywords related to cleanliness.
# - Serv: Keywords related to service.
# - Fac: Keywords related to facilities.
# - VfM: Keywords related to value for money.

# Only return keywords for each category as a list. If there are no relevant keywords, return an empty list.

# Full Text: "{full_text}"

# Extracted Keywords:
# - Loc: 
# - Clean:
# - Serv:
# - Fac:
# - VfM:
# """

# # 입력 변수를 사용하여 프롬프트 템플릿 생성
# template = PromptTemplate(template=prompt_template, input_variables=["full_text"])

# # 주어진 API 키와 모델 이름으로 OpenAI LLM 초기화
# llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo")
# llm_chain = LLMChain(llm=llm, prompt=template)

# # LLM 결과를 저장할 리스트 초기화
# results = []

# # 원본 리뷰 데이터를 기반으로 GPT 출력 저장
# for index, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Processing Reviews"):
#     full_text = row['Full']  # 'Full' 열 참조
#     result = llm_chain.run({"full_text": full_text})
    
#     results.append(result)

# # GPT 출력 결과를 새로운 열로 추가
# df_test.loc[:, 'LLM_result'] = results

# # 작업 중간 결과를 파일로 저장
# df_test.to_csv("keyword_extraction_test.csv", index=False, encoding='utf-8-sig')

# # 최종 결과를 파일로 저장
# df_test.to_csv("keyword_extraction_final_test.csv", index=False, encoding='utf-8-sig')

# # 결과 확인
# print(df_test[['Full', 'LLM_result']].head(20))
# %% 나중에 파싱

# import pandas as pd
# import ast

# # LLM_result 열이 포함된 데이터프레임을 불러온 후, 파싱 작업을 수행합니다.
# df_test = pd.read_csv("keyword_extraction_final_test.csv")

# # 각 키워드를 저장할 새로운 열을 추가합니다.
# df_test['Loc_keywords'] = None
# df_test['Clean_keywords'] = None
# df_test['Serv_keywords'] = None
# df_test['Fac_keywords'] = None
# df_test['VfM_keywords'] = None

# # LLM_result에서 키워드를 추출하여 각 열에 저장합니다.
# for index, row in df_test.iterrows():
#     result = row['LLM_result']
    
#     try:
#         # 각 카테고리의 키워드를 추출
#         loc_kw = result.split("Loc:")[1].split("Clean:")[0].strip()
#         clean_kw = result.split("Clean:")[1].split("Serv:")[0].strip()
#         serv_kw = result.split("Serv:")[1].split("Fac:")[0].strip()
#         fac_kw = result.split("Fac:")[1].split("VfM:")[0].strip()
#         vfm_kw = result.split("VfM:")[1].strip()
#     except (IndexError, ValueError) as e:
#         # 만약 파싱에 실패하면 빈 문자열로 처리
#         loc_kw, clean_kw, serv_kw, fac_kw, vfm_kw = "", "", "", "", ""
    
#     # 각 키워드를 데이터프레임의 해당 열에 저장
#     df_test.at[index, 'Loc_keywords'] = loc_kw
#     df_test.at[index, 'Clean_keywords'] = clean_kw
#     df_test.at[index, 'Serv_keywords'] = serv_kw
#     df_test.at[index, 'Fac_keywords'] = fac_kw
#     df_test.at[index, 'VfM_keywords'] = vfm_kw

# # 결과를 파일로 저장
# df_test.to_csv("keyword_extraction_parsed.csv", index=False, encoding='utf-8-sig')

# # 결과 확인
# print(df_test[['Loc_keywords', 'Clean_keywords', 'Serv_keywords', 'Fac_keywords', 'VfM_keywords']].head(20))
