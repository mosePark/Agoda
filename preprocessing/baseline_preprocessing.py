'''
추가 데이터랑 호텔 445개 랜덤 데이터랑 합쳤음
초기 GOSDT 적합하기전 데이터 샘플 만드는 작업
'''


import numpy as np
import pandas as pd

import openpyxl
import re
import time
import pathlib

from sklearn.ensemble import GradientBoostingClassifier

from gosdt import GOSDT
from gosdt.model.threshold_guess import compute_thresholds

def extract_scores(text):
    
    scores = re.findall(r'\d+\.?\d*', text)
    return scores

def find_closest(number, numbers):
    return min(numbers, key=lambda x: abs(x - number))


# df = pd.read_excel("C:/Users/UOS/proj_0/GOSDT/true_pred_3-type-Q.xlsx", index_col=0)



# df.isnull().sum()

# df['y_hat']

# num = 0

# max_ = []
# score = []
# min_ = []
# for txt in df['y_hat'] :

#     str_list = extract_scores(txt)
#     float_list = [float(item) for item in str_list]
#     float_list = np.sort(float_list)

#     if len(float_list) == 3 :
#         max_.append(float_list[2])
#         score.append(float_list[1])
#         min_.append(float_list[0])

#     else :
#         max_.append("?")
#         score.append("?")
#         min_.append("?")

# df['max_y_hat'] = max_
# df['med_y_hat'] = score
# df['min_y_hat'] = min_

# df.to_excel("전처리.xlsx")

df = pd.read_excel("C:/Users/UOS/proj_0/GOSDT/gosdt-guesses/baseline/전처리.xlsx", index_col=0)

'''
결측
- 국가 : 18
- 룸 유형 : 127
- 머무른 기간 : 10
- 제목 : 1
'''

df_2 = pd.read_csv("C:/Users/UOS/proj_0/data/agoda/another/another-data.csv", index_col=0, encoding = 'utf-8-sig')
df_2.columns

df_2.isnull().sum()

df_2 = df_2.drop(['rating', 'Fac_score', 'VfM_score'], axis=1) # 잘못 불러온 칼럼 삭제

df_2.rename(
    columns={
        'Fac_Score' : 'Fac_score',
        'VfM_Score' : 'VfM_score'
    }, inplace=True
)

df_2 = df_2.reindex(columns=['hotel_name', 'Address', 'Loc_score', 'Clean_score', 'Serv_score', 'Fac_score', 'VfM_score', 'Rating', 'url'])

'''
rating 숫자 처리
'''
ratings = []
for tx in df_2['Rating'] :
    msg = tx.split()[0]
    ratings.append(msg)

df_2['Rating'] = ratings

df_2['Rating'].value_counts() # 0점은 적절히 대치법으로 수행

df_2.columns

df_2['Rating'] = df_2['Rating'].astype(float)

nan_idx = df_2[df_2['Rating'] == 0].index



# pd.DataFrame([r1, r2]).transpose().to_excel('점수비교.xlsx')

'''
점수 값 대치
'''
for i in nan_idx :
    s = (df_2['Loc_score'][i] + df_2['Clean_score'][i] + df_2['Serv_score'][i] + df_2['Fac_score'][i] + df_2['VfM_score'][i])/5/2
    rating_list = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    closest = find_closest(s, rating_list)
    df_2['Rating'][i] = closest

df_2['Rating'].value_counts()



'''
데이터 합치기(호텔데이터+추가데이터)
'''
df

df_2

base_data = pd.merge(df, df_2, on='hotel_name')

# base_data.to_excel('base.xlsx')

'''
Stay Duration 전처리
머무른기간 : `Duration`
숙박이용 월 : `Stay_month`
숙박이용 연도 : `Stay_year`
'''

base_data.isnull().sum()

base_data['Duration'] = base_data['Stay Duration'].str.extract(r'Stayed (\d+) nights').astype('Int64')
base_data['Stay_month'] = base_data['Stay Duration'].str.extract(r'in (\w+)')
base_data['Stay_year'] = base_data['Stay Duration'].str.extract(r'(\d{4})').astype('Int64')

'''
Date 전처리
리뷰작성 연도 : `Review_year`
리뷰작성 월 : `Review_month`
리뷰작성 일 : `Review_day`
'''

base_data['Review_year'] = pd.to_numeric(base_data['Date'].str.extract(r'(\d{4})')[0], errors='coerce')
base_data['Review_month'] = base_data['Date'].str.extract(r'Reviewed (\w+)')[0]
base_data['Review_day'] = pd.to_numeric(base_data['Date'].str.extract(r'(\d{1,2}),')[0], errors='coerce')


'''
Room type 전처리

'''
room_type = base_data['Room Type']

# 소문자화, 비문자를 공백으로 대체
room_type_cleaned = [re.sub(r'\W+', ' ', str(rt).lower()) if pd.notnull(rt) else '' for rt in room_type]

# 단어 분리
words = ' '.join(room_type_cleaned).split()

# 단어 빈도
from collections import Counter
word_freq = Counter(words)

word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
word_freq_df.to_excel("word table.xlsx", index = False)

priority_keyword = ['deluxe', 'king', 'queen', 'double', 'guest', 'standard', 'studio', 'suite', 'twin', 'triple', 'family']
hot_keyword = ['non-smoking', 'view']

room_type_keyword = []
for wds in room_type_cleaned:
    room_keywords = []
    for key in priority_keyword:
        if key in wds:
            room_keywords.append(key)
            break

    # 최종 선택된 키워드를 저장
    room_type_keyword.append(' '.join(room_keywords))

# 결과를 DataFrame에 추가
base_data['room_type'] = room_type_keyword

non_smoking = []
for wds in room_type_cleaned :
    if 'smoking' in wds :
        non_smoking.append(1)
    else :
        non_smoking.append(0)

base_data['Non_smoking'] = non_smoking

views = []
for wds in room_type_cleaned :
    if 'view' in wds :
        views.append(1)
    else :
        views.append(0)

base_data['View'] = views

base_data.to_excel("base_data.xlsx")
