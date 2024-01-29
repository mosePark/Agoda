import os

import numpy as np
import pandas as pd

''' folder list
active-batch
active-batch-2
active-batch-3
active-batch-passive

another
'''

'''
load_csv_files() : 특정 위치 폴더 내 .csv파일 읽는 함수
deudpe() : 데이터 내 중복을 찾고 데이터를 출력하는 함수
'''

def load_csv_files(folder_path):
    # 폴더 내의 모든 파일 목록 가져오기
    files = os.listdir(folder_path)

    # CSV 파일만을 필터링합니다.
    csv_files = [file for file in files if file.endswith('.csv')]

    # 모든 CSV 파일을 DataFrame으로 읽어들여 리스트에 저장
    dataframes = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, index_col=0)
        dataframes.append(df)

    return dataframes

def dedupe(df) :
    # 중복된 행 찾기
    duplicates = df[df.duplicated()]
    df_no_duplicates = df.drop_duplicates()

    return df_no_duplicates

'''data load
df_1 : batch process
df_2 : batch process2
df_3 : batch process3
df_4 : passive
'''

folder_path = 'C:/Users/UOS/proj_0/data/agoda/active-batch'
csv_dataframes = load_csv_files(folder_path)

df_1 = pd.concat(csv_dataframes, axis=0).reset_index(drop=True)

# =====

folder_path = 'C:/Users/UOS/proj_0/data/agoda/active-batch-2'
csv_dataframes = load_csv_files(folder_path)

df_2 = pd.concat(csv_dataframes, axis=0).reset_index(drop=True)

# =====

folder_path = 'C:/Users/UOS/proj_0/data/agoda/active-batch-3'
csv_dataframes = load_csv_files(folder_path)

df_3 = pd.concat(csv_dataframes, axis=0).reset_index(drop=True)

# =====

folder_path = 'C:/Users/UOS/proj_0/data/agoda/active-batch-passive'
csv_dataframes = load_csv_files(folder_path)

df_4 = pd.concat(csv_dataframes, axis=0).reset_index(drop=True)


''' data merge
df = df_1 + df_2 + df_3 + df_4

'''
df = pd.concat([df_1, df_2, df_3, df_4], axis=0).reset_index(drop=True)
# df.drop(df.columns[-1], axis=1, inplace=True) # 리뷰페이지 제거

df = dedupe(df)


df.to_excel("전체데이터.xlsx", index = False)
