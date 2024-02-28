'''
초기 GOSDT 적합하기전 데이터 샘플 만드는 작업
text imbedding 작업

dimension log
- 1536 (default) : GOSDT fitting시 kernel 터짐

'''

import numpy as np
import pandas as pd
import requests
import openpyxl

from openai import OpenAI

'''
dimension 바꿔서 요청하기
'''
# # Request
# api_url = "https://api.openai.com/v1/embeddings"
# headers = {
#     "Authorization": "Bearer your-api-key!",
#     "Content-Type": "application/json"
# }
# data = {
#     "input": "The food was delicious and the waiter...",
#     "model": "text-embedding-3-small",
#     "encoding_format": "float",
#     "dimensions" : 100
# }

# response = requests.post(api_url, headers=headers, json=data)
# print(response.json())
# response.json()['data'][0]['embedding'] # embedding vector 출력

df = pd.read_excel("/.../base_data.xlsx", index_col=0)

def get_embedding(text, model="text-embedding-3-small", dimensions=100):
   api_url = "https://api.openai.com/v1/embeddings"
   headers = {
       "Authorization": "Bearer your-api-key",
       "Content-Type": "application/json",
       "User-Agent" : "..."

   }
   data = {
       "input": text,
       "model": model,
       "encoding_format": "float",
       "dimensions": dimensions
   }

   response = requests.post(api_url, headers=headers, json=data)
   embedding_vector = response.json()['data'][0]['embedding']
   return embedding_vector


df['Embedding'] = df['Text'].apply(lambda x: get_embedding(x, model='text-embedding-3-small', dimensions=100))

df.to_excel('baseline_embedding.xlsx', index=False)

df['Embedding'].to_excel('embedding.xlsx')



# ===== 임베딩 전처리 작업
ebd = pd.DataFrame(df['Embedding'].to_list())

ebd.to_csv('embedding.csv')
ebd.to_excel('embedding.xlsx')
