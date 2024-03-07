import numpy as np
import pandas as pd

import os
import re

def find_closest(number, numbers):
    return min(numbers, key=lambda x: abs(x - number))

os.chdir('.../Agoda-Data/selenium/another')

df = pd.read_csv("another-data.csv", index_col=0)
df.head()


df.isnull().sum()

# 잘못 불러온 칼럼 삭제
df = df.drop(['rating', 'Fac_score', 'VfM_score'], axis=1)

df.rename(
    columns={
        'Fac_Score' : 'Fac_score',
        'VfM_Score' : 'VfM_score'
    }, inplace=True
)


df = df.reindex(columns=['hotel_name', 'Address', 'Loc_score', 'Clean_score', 'Serv_score', 'Fac_score', 'VfM_score', 'Rating', 'url'])

'''
rating 숫자 처리
'''
ratings = []
for tx in df['Rating'] :
    msg = tx.split()[0]
    ratings.append(msg)

df['Rating'] = ratings
df['Rating'].value_counts() # 0점은 적절히 대치법으로 수행
df['Rating'] = df['Rating'].astype(float)

nan_idx = df[df['Rating'] == 0].index



'''
점수 값 대치
'''
for i in nan_idx :
    s = (df['Loc_score'][i] + df['Clean_score'][i] + df['Serv_score'][i] + df['Fac_score'][i] + df['VfM_score'][i])/5/2
    rating_list = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    closest = find_closest(s, rating_list)
    df['Rating'][i] = closest

df['Rating'].value_counts()

df[df['Loc_score'].isnull()]['hotel_name']

df.to_csv('another.csv', encoding='utf-8-sig')
