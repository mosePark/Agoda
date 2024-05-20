'''
그때그때 계산하기 너무 귀찮아서
거리행렬을 그냥 반환하는 것으로 작성해서
저장 후 그때그때 사용하기
'''

import os

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import hamming

# 해밍 거리 계산 함수
def cal_ham_dist(matrix):
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix  # 희소 행렬을 배열로 변환
    num_samples = matrix.shape[0]
    distances = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            dist = hamming(matrix[i], matrix[j])
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

# 작업 디렉토리 설정
os.chdir('C:/Users/mose/agoda/data/')

# 데이터 로드
eng = pd.read_csv("eng.csv", index_col=0)

# 벡터화
cntvectorizer = CountVectorizer(stop_words='english')
cnt_X = cntvectorizer.fit_transform(eng['Text'])

tfidfvectorizer = TfidfVectorizer(stop_words='english')
tf_X = tfidfvectorizer.fit_transform(eng['Text'])

# 해밍 거리 행렬 계산
cnt_ham_distances = cal_ham_dist(cnt_X)
tf_ham_distances = cal_ham_dist(tf_X)

# # 거리 행렬 저장
# np.save('cnt_ham_distances.npy', cnt_ham_distances)
# np.save('tf_ham_distances.npy', tf_ham_distances)
