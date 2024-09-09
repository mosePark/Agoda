'''
temperature 별 sim 계산 차이 식별
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

os.chdir('C:/Users/mose/agoda/data/')  # home
# os.chdir('C:/Users/UOS/Desktop/Agoda-Data/raw')  # lab

df = pd.read_csv("agoda2.csv")

#%%
df.isnull().sum()

#%%
# 랜덤 시드 설정
random_state = 240902
np.random.seed(random_state)

# 데이터프레임의 인덱스에서 무작위로 3000개의 인덱스 선택
random_indices = np.random.choice(df.index, size=3000, replace=False)

# 선택된 인덱스를 사용하여 새로운 데이터프레임 생성
df = df.loc[random_indices]

#%%
# 선택된 인덱스를 txt 파일로 저장
output_path = 'random_indices.txt'

with open(output_path, 'w') as f:
    for index in random_indices:
        f.write(f"{index}\n")

#%%
'''
Prompt 1 - 원본 리뷰를 기반으로 유사한 문맥의 리뷰를 생성
'''
prompt_template = """
Read the following review text and generate another review text with a similar context:

Note: The original text is composed in the form of title + "\n" + body, as shown below:
Original Review: "{Full}"

Please generate a review in a similar format, with a title and a body:

Generated Review:
"""

# 입력 변수를 사용하여 프롬프트 템플릿 생성
template = PromptTemplate(template=prompt_template, input_variables=["Full"])

# 주어진 API 키와 모델 이름으로 OpenAI LLM 초기화
llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo", temperature=1.0)
llm_chain = LLMChain(llm=llm, prompt=template)

# 생성된 리뷰를 저장할 리스트 초기화
generated_reviews = []

# 원본 리뷰 데이터를 기반으로 비슷한 문맥의 리뷰 생성 (gen1)
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Reviews"):
    review = row['Full']  # 'Text' 열 참조
    result = llm_chain.run({"Full": review})
    generated_review = result.strip()
    generated_reviews.append(generated_review)
    
    # 생성된 리뷰를 데이터프레임의 해당 행에 추가
    df.at[index, 'generated_review'] = generated_review
    
    # 현재 생성된 리뷰를 파일로 저장
    df.to_csv("gen_1.csv", encoding='utf-8-sig')

# 최종 생성된 리뷰를 파일로 안전하게 저장
df.to_csv("gen_1.csv", encoding='utf-8-sig')

#%%
'''
Prompt 2 - 두 번째 유사 리뷰 생성
'''
llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo", temperature=1.0)  # temperature를 1.0로 설정
llm_chain = LLMChain(llm=llm, prompt=template)

generated_reviews_2 = []

# gen1 리뷰 데이터를 기반으로 두 번째 유사 리뷰 생성 (gen2)
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Reviews"):
    review = row['Full']  # 'Text' 열 참조
    result = llm_chain.run({"Full": review})
    generated_review = result.strip()
    generated_reviews_2.append(generated_review)
    
    # 생성된 리뷰를 데이터프레임의 해당 행에 추가
    df.at[index, 'generated_review_2'] = generated_review
    
    # 현재 생성된 리뷰를 파일로 저장
    df.to_csv("gen_2.csv", encoding='utf-8-sig')

# 최종 생성된 리뷰를 파일로 안전하게 저장
df.to_csv("gen_2.csv", encoding='utf-8-sig')

#%%
'''
Prompt 3 - gen1을 입력으로 하여 gen1-2 생성
'''

prompt_template_3 = """
Read the following generated review text and generate another review text with a similar context:

Note: The original text is composed in the form of title + "\n" + body, as shown below:
Original Generated Review: "{generated_review}"

Please generate another review in a similar format, with a title and a body:

Generated Review 2:
"""

# 입력 변수를 사용하여 프롬프트 템플릿 생성
template_3 = PromptTemplate(template=prompt_template_3, input_variables=["generated_review"])

llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo", temperature=1.0)  # temperature를 1.0으로 설정
llm_chain_3 = LLMChain(llm=llm, prompt=template_3)

generated_reviews_3 = []

# gen1을 입력으로 받아서 gen1-2 생성
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Reviews"):
    generated_review = row['generated_review']  # 'generated_review' 열 참조 (gen1 리뷰)
    result = llm_chain_3.run({"generated_review": generated_review})
    generated_review_3 = result.strip()
    generated_reviews_3.append(generated_review_3)
    
    # 생성된 리뷰를 데이터프레임의 해당 행에 추가
    df.at[index, 'generated_review_3'] = generated_review_3  # 'generated_review_3' 열에 저장
    
    # 현재 생성된 리뷰를 파일로 저장
    df.to_csv("gen_1_2.csv", encoding='utf-8-sig')

# 최종 생성된 리뷰를 파일로 안전하게 저장
df.to_csv("gen_1_2.csv", encoding='utf-8-sig')
