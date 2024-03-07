'''

task1 : agoda.csv + gpt prediction + another data.csv

task2 : missing value, dummy variable, one hot encoding

task3 : extract score from y_hat : extract_score(text)

'''

def extract_score(text):
    # 입력이 문자열이 아니거나 NaN/None일 경우 처리
    if pd.isnull(text) or not isinstance(text, str):
        return None
    
    # 정규식을 사용하여 부동 소수점 형식의 숫자 패턴을 찾음
    pattern = r'\d+(\.\d+)?'
    match = re.search(pattern, text)
    
    if match:
        # 숫자가 발견되면 해당 숫자를 반환
        return float(match.group())
    else:
        # 숫자가 없을 경우 None 반환
        return None



import numpy as np
import pandas as pd

import os
import re
from collections import Counter

os.chdir('C:/Users/UOS/Desktop/Agoda-Data/raw')

agoda = pd.read_csv("agoda+y_hat.csv", index_col=0)
agoda.head()

another = pd.read_csv("another.csv", index_col=0)
another.head()



drop_hotel = ['Luxury 3BR Duplex w Private Patio in Upper East',
    'TOWNHOUSE 222',
    # 'Casa Cipriani New York',
    'Hampton Inn New York Times Square'
]

print("총 호텔 수는 : ", len(another))

for i in range(len(drop_hotel)) :
    df = df[df['hotel_name'] != drop_hotel[i]]
    another = another[another['hotel_name'] != drop_hotel[i]]

print("총 호텔 수는 : ", len(another))

another[another['hotel_name'] == 'Casa Cipriani New York']

another.loc[another['Loc_score'].isna(), 'Loc_score'] = 8.6
another.loc[another['Clean_score'].isna(), 'Clean_score'] = 9.8
another.loc[another['Serv_score'].isna(), 'Serv_score'] = 9.4
another.loc[another['Fac_score'].isna(), 'Fac_score'] = 9.5
another.loc[another['VfM_score'].isna(), 'VfM_score'] = 8.3

'''
데이터 합치기(호텔데이터+추가데이터)
'''

df = pd.merge(agoda, another, on='hotel_name')

'''
Stay Duration 전처리
머무른기간 : `Duration`
숙박이용 월 : `Stay_month`
숙박이용 연도 : `Stay_year`
'''

df['Duration'] = df['Stay Duration'].str.extract(r'Stayed (\d+) night[s]?').astype('Int64')
df['Stay_month'] = df['Stay Duration'].str.extract(r'in (\w+)')
df['Stay_year'] = df['Stay Duration'].str.extract(r'(\d{4})').astype('Int64')

'''
Date 전처리
리뷰작성 연도 : `Review_year`
리뷰작성 월 : `Review_month`
리뷰작성 일 : `Review_day`
'''

df['Review_year'] = pd.to_numeric(df['Date'].str.extract(r'(\d{4})')[0], errors='coerce')
df['Review_month'] = df['Date'].str.extract(r'Reviewed (\w+)')[0]
df['Review_day'] = pd.to_numeric(df['Date'].str.extract(r'(\d{1,2}),')[0], errors='coerce')

'''

room type 단어 빈도표

'''
room_type = df['Room Type']

# 소문자화, 비문자를 공백으로 대체
room_type_cleaned = [re.sub(r'\W+', ' ', str(rt).lower()) if pd.notnull(rt) else '' for rt in room_type]

# 단어 분리
words = ' '.join(room_type_cleaned).split()

# 단어 빈도
word_freq = Counter(words)

word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
word_freq_df.to_excel("전체 데이터의 룸타입 단어 빈도표.xlsx", index = False)

# # 기존방식

# priority_keyword = ['deluxe', 'king', 'queen', 'double', 'guest', 'standard', 'studio', 'suite', 'twin', 'triple', 'family']
# hot_keyword = ['non-smoking', 'view']

# room_type_keyword = []
# for wds in room_type_cleaned:
#     room_keywords = []
#     for key in priority_keyword:
#         if key in wds:
#             room_keywords.append(key)
#             break

#     # 최종 선택된 키워드를 저장
#     room_type_keyword.append(' '.join(room_keywords))

# # 결과를 DataFrame에 추가
# df['room_type'] = room_type_keyword

# non_smoking = []
# for wds in room_type_cleaned :
#     if 'smoking' in wds :
#         non_smoking.append(1)
#     else :
#         non_smoking.append(0)

# df['Non_smoking'] = non_smoking

# views = []
# for wds in room_type_cleaned :
#     if 'view' in wds :
#         views.append(1)
#     else :
#         views.append(0)

# df['View'] = views

'''

room type 주요 키워드 one hot encoding

'''

keywords = ['deluxe', 'king', 'queen', 'double',
            'guest', 'standard', 'studio', 'suite',
            'twin', 'triple', 'family', 'smoking',
            'view']

for keyword in keywords:
    df[keyword] = [1 if keyword in room_type else 0 for room_type in room_type_cleaned]


y_hat_value = []

for i in range(len(df)) :
    y_hat_value.append(extract_score(df['y_hat'][i]))

df['y_hat_'] = y_hat_value



df.to_csv("train.csv", encoding='utf-8-sig')
