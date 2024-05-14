'''
50개 이웃 추출
tf-idf vs cnt
'''


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

from numpy import dot
from numpy.linalg import norm

import pandas as pd

import os

os.chdir('C:/Users/mose/agoda/data/')

eng = pd.read_csv("eng.csv", index_col=0)

eng['difference'].value_counts()

# 데이터 로드
eng = pd.read_csv("eng.csv", index_col=0)

# 벡터화
cntvectorizer = CountVectorizer(stop_words='english')
cnt_X = cntvectorizer.fit_transform(eng['Text'])

tfidfvectorizer = TfidfVectorizer(stop_words='english')
tf_X = tfidfvectorizer.fit_transform(eng['Text'])

# 코사인 유사도 계산 함수
def get_cosine_similarity(matrix):
    norms = np.linalg.norm(matrix, axis=1)
    zero_norms = norms == 0
    safe_denom = np.outer(norms, norms)
    safe_denom[zero_norms, :] = 1
    safe_denom[:, zero_norms] = 1
    cosine_similarity_matrix = np.dot(matrix, matrix.T) / safe_denom
    cosine_similarity_matrix[zero_norms, :] = 0
    cosine_similarity_matrix[:, zero_norms] = 0
    np.fill_diagonal(cosine_similarity_matrix, 0)
    return cosine_similarity_matrix

# 유클리디안 거리 계산 함수
def get_euclidean_distance(matrix):
    euclidean_distance_matrix = cdist(matrix, matrix, metric='euclidean')
    np.fill_diagonal(euclidean_distance_matrix, np.inf)
    return euclidean_distance_matrix

# 유사도 및 거리 행렬 계산
tf_cosine_similarity_matrix = get_cosine_similarity(tf_X.toarray())
cnt_cosine_similarity_matrix = get_cosine_similarity(cnt_X.toarray())

tf_euclidean_distance_matrix = get_euclidean_distance(tf_X.toarray())
cnt_euclidean_distance_matrix = get_euclidean_distance(cnt_X.toarray())

# 가장 가까운 이웃 50개 추출 함수
def get_top_n_neighbors(similarity_or_distance_matrix, n=50, largest=True):
    if largest:
        top_n_indices = np.argpartition(similarity_or_distance_matrix, -n, axis=1)[:, -n:]
    else:
        top_n_indices = np.argpartition(similarity_or_distance_matrix, n, axis=1)[:, :n]
    return top_n_indices

# 각 임베딩 방법에 따른 가장 가까운 50개의 이웃 인덱스 추출
tf_top_50_cosine_indices = get_top_n_neighbors(tf_cosine_similarity_matrix, n=50, largest=True)
cnt_top_50_cosine_indices = get_top_n_neighbors(cnt_cosine_similarity_matrix, n=50, largest=True)

tf_top_50_euclidean_indices = get_top_n_neighbors(tf_euclidean_distance_matrix, n=50, largest=False)
cnt_top_50_euclidean_indices = get_top_n_neighbors(cnt_euclidean_distance_matrix, n=50, largest=False)

# 특정 인덱스에 대한 이웃 비교 및 유사도 계산 함수
def calculate_neighbor_similarity(idx, neighbors_1, neighbors_2):
    set_1 = set(neighbors_1[idx])
    set_2 = set(neighbors_2[idx])
    intersection = len(set_1 & set_2)
    union = len(set_1 | set_2)
    similarity = intersection / union if union != 0 else 0
    return similarity

# 유사도 값 계산
tf_cnt = [calculate_neighbor_similarity(i, tf_top_50_cosine_indices, cnt_top_50_cosine_indices) for i in range(len(eng))]

# tf_cnt를 DataFrame으로 변환하여 eng.index와 매칭
tf_cnt_df = pd.DataFrame({'similarity': tf_cnt}, index=eng.index)

# 결과 출력
print(tf_cnt_df)

# 0보다 큰 값의 인덱스와 그 값을 출력하고 총 개수 반환
count = 0
for index, value in tf_cnt_df['similarity'].items():
    if value > 0:
        print(f"Index: {index}, Value: {value}")
        count += 1

print(f"Total count of indices with values greater than 0: {count}")

# 빈도표
value_counts = tf_cnt_df['similarity'].value_counts().reset_index()
value_counts.columns = ['value', 'frequency']
print(value_counts)

# 유사도 값이 1인 데이터의 인덱스 찾기
indices_with_similarity_1 = tf_cnt_df[tf_cnt_df['similarity'] == 1].index

# eng DataFrame에서 해당 인덱스의 'Text' 열 출력
texts_with_similarity_1 = eng.loc[indices_with_similarity_1, 'Text']

# 결과 출력
print(texts_with_similarity_1)

value_counts.to_excel("50_value_counts.xlsx")
tf_cnt_df.to_excel("50_tf_cnt_output.xlsx")

# 도수분포표 생성: 0.1 단위 구간
bins = np.arange(0, 1.1, 0.1)
labels = [f'{round(bins[i], 1)}-{round(bins[i+1], 1)}' for i in range(len(bins)-1)]
tf_cnt_df['binned'] = pd.cut(tf_cnt_df['similarity'], bins=bins, labels=labels, include_lowest=True)

# 도수분포표 계산
frequency_distribution = tf_cnt_df['binned'].value_counts().sort_index().reset_index()
frequency_distribution.columns = ['interval', 'frequency']

# 도수분포표 출력
print(frequency_distribution)
