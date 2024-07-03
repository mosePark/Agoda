import os
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

'''
API 로드
'''

# API KEY 정보 로드
current_directory = os.getcwd()
project_root = current_directory
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)

# API 키를 환경 변수에서 가져오기
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY in your .env file.")

'''
데이터 로드
'''

os.chdir('C:/Users/mose/agoda/data/')
df = pd.read_csv("agoda.csv")

'''
Prompt
'''

# 점수 예측을 위한 프롬프트 템플릿을 정의합니다.
prompt_template = """
어떤 유저의 국적, 여행객 유형에 대한 정보와 유저가 작성한 리뷰 텍스트를 갖고 10점 만점에 몇점을 줬을지 예측해줘. 단, 소수점 첫째자리까지:

Review: "{review}"
Country: "{country}"
Traveler Type: "{traveler_type}"
Score (1-10):
"""

# 템플릿을 사용하여 프롬프트 생성
template = PromptTemplate(template=prompt_template, input_variables=["review", "country", "traveler_type"])

# OpenAI LLM과 프롬프트 템플릿을 사용하여 LLMChain을 만듭니다.
llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo")
llm_chain = LLMChain(llm=llm, prompt=template)

# 예측 결과를 저장할 리스트 초기화
predicted_scores = []

# 리뷰 데이터와 국가 정보를 입력하고 점수를 예측합니다.
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Reviews"):
    review = row['Text']
    country = row['Country']
    traveler_type = row['Traveler Type']
    result = llm_chain.run({"review": review, "country": country, "traveler_type": traveler_type})
    predicted_score = result.strip()
    predicted_scores.append(predicted_score)
    
    # 예측 결과를 데이터프레임의 해당 행에 추가
    df.at[index, 'y_hat__'] = predicted_score
    
    # 현재까지의 예측 결과를 파일로 저장
    df.to_csv("국가+리뷰+여행객 예측.csv", index=False, encoding='utf-8-sig')

# 최종 예측 결과를 파일로 저장 (안전하게)
df.to_csv("국가+리뷰+여행객 예측 최종.csv", index=False, encoding='utf-8-sig')
