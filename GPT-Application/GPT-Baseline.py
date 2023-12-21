import os
import openai
from langchain.chat_models import ChatOpenAI

open_api_key = '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'
chat_model = ChatOpenAI(openai_api_key = open_api_key, verbose = False)

content = '이 리뷰를 보고 몇점을 줬을지 예측해줘. 리뷰는 다음과 같아. "Bottom of the 18 meter pool was too scary. Staff were nice though."'
result = chat_model.predict(content)

print(result)