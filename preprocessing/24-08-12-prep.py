'''
대대적인 전처리 작업 시작

작업 목록
1. title, text " 제거
2. title, text 합치는 text로
3. 결측치, 중복행 제거
4. 카테고리 점수 데이터 (another.csv) 과 병합
5. 이외 추가 결측 제거
'''

#%%
def remove_trailing_quote(text):
    if isinstance(text, str):
        return text.rstrip('”')
    return text


#%%
import re
import os
import numpy as np
import pandas as pd

os.chdir('C:/Users/UOS/proj_0/preprocessing/')

df = pd.read_csv("agoda.csv")

df.rename(columns={'hotel_name': 'Hotel'}, inplace=True)

df.isnull().sum()


#%% 작업 1

df['Text'] = df['Text'].fillna('')

df['Title'] = df['Title'].apply(remove_trailing_quote)
df['Text'] = df['Text'].apply(remove_trailing_quote)

df.head()

#%% 작업 2

df['Full'] = df['Title'].fillna('') + "\n" + df['Text'].fillna('')
df['Full'] = df['Full'].replace("\n", np.nan)

df = df.dropna(subset=['Full']) # idx = 1945 제거, FUll = NaN
#%% 작업 3
 
# 중복 데이터 (행) 조회 및 제거
df_dup = df[df.duplicated(subset=['Hotel', 'Score','Country', 'Traveler Type', 'Stay Duration', 'Title', 'Text'], keep=False)]
df_dup

# 중복된 행 중 첫 번째 행의 인덱스 저장
first_indices = df_dup.index

# 중복된 행 제거하고 각 그룹의 첫 번째 행만 남기기
df_unique = df.drop_duplicates(subset=['Hotel', 'Score', 'Country', 'Traveler Type', 'Stay Duration', 'Title', 'Text'], keep='first')

df_unique[df_unique['Full'].isnull()] # 1945 제목, 텍스트 둘다 없음

# 특정 인덱스(예: 1945) 제거
df = df_unique.drop(index=1945, errors='ignore')

df.isnull().sum()

ctry_idx = df[df['Country'].isnull()].index

#%% 작업 3-2 (국가 결측 언어 감지)

import fasttext
import pycountry
import requests

# url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
# output_path = "lid.176.bin"

# # 파일 다운로드
# response = requests.get(url, stream=True)
# with open(output_path, "wb") as file:
#     for chunk in response.iter_content(chunk_size=8192):
#         if chunk:
#             file.write(chunk)

# print(f"다운로드 완료: {output_path}")


# fastText 모델 로드
model = fasttext.load_model('lid.176.bin')

# 언어 코드와 국가 매핑
language_to_country = {
    'en': 'US', 'tl': 'PH', 'id': 'ID', 'ko': 'KR', 'th': 'TH', 'ms': 'MY', 'hi': 'IN',
    'zh': 'CN', 'ar': 'SA', 'ru': 'RU', 'es': 'ES', 'uk': 'UA', 'vi': 'VN', 'fr': 'FR',
    'de': 'DE', 'ja': 'JP', 'he': 'IL', 'sv': 'SE', 'no': 'NO', 'da': 'DK', 'nl': 'NL',
    'it': 'IT', 'pl': 'PL', 'cs': 'CZ', 'tr': 'TR', 'fi': 'FI', 'ur': 'PK', 'bn': 'BD',
    'pt': 'BR', 'mt': 'MT', 'et': 'EE', 'my': 'MM', 'az': 'AZ', 'el': 'GR'
}

# 결측값 대체
for index, row in df.iterrows():
    if pd.isnull(row['Country']):
        # 'Text'의 언어 감지
        prediction = model.predict(row['Text'])[0][0].replace('__label__', '')
        # 감지된 언어를 기반으로 국가를 매핑하여 결측값 대체
        df.at[index, 'Country'] = language_to_country.get(prediction, 'Unknown')


df = df[df['Country']!='Unknown'] # Unknown 지우기

df.isnull().sum()
#%% another 데이터와 병합

adf = pd.read_csv("another.csv", index_col=0)

adf.columns, df_unique.columns

len(adf), len(df_unique)

df = df.merge(adf, how='left', left_on='Hotel', right_on='hotel_name')
df = df.drop(columns=['hotel_name'])


df.isnull().sum()

# 카테고리 점수 결측 29개 조회
df[df[['Loc_score', 'Clean_score', 'Serv_score', 'Fac_score', 'VfM_score']].isnull().any(axis=1)]

# 각 칼럼의 결측값 인덱스 가져오기
loc_missing_idx = df[df['Loc_score'].isnull()].index
clean_missing_idx = df[df['Clean_score'].isnull()].index
serv_missing_idx = df[df['Serv_score'].isnull()].index
fac_missing_idx = df[df['Fac_score'].isnull()].index
vfm_missing_idx = df[df['VfM_score'].isnull()].index

# 모든 결측값 인덱스가 동일한지 확인
all_missing_idx_equal = (loc_missing_idx.equals(clean_missing_idx) and
                         clean_missing_idx.equals(serv_missing_idx) and
                         serv_missing_idx.equals(fac_missing_idx) and
                         fac_missing_idx.equals(vfm_missing_idx))

# 결과 출력
print("All indices are equal:", all_missing_idx_equal)

if all_missing_idx_equal:
    print("The indices of missing values in all five columns are identical.")
else:
    print("The indices of missing values in the five columns are not identical.")

# 카테고리 점수 결측 제거

df = df.dropna(subset=['Loc_score'])

df.isnull().sum()

df.info()

df_reset = df.reset_index(drop=True)


# %%

df.to_csv("agoda2-original-idx.csv", encoding='utf-8-sig', index=False)
df_reset.to_csv("agoda2.csv", encoding='utf-8-sig', index=False)
