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

def nbds_sampling(distances, top_n):
    """
    주어진 거리 행렬에서 각 행의 가장 가까운 이웃 상위 n개를 추출합니다.
    
    Args:
        distances (numpy.ndarray): 거리 행렬
        top_n (int): 추출할 이웃의 개수
    
    Returns:
        List[numpy.ndarray]: 각 행에 대한 상위 n개 이웃의 인덱스를 포함한 리스트
    """
    neighbors = []
    for i in range(distances.shape[0]):
        nearest_indices = np.argsort(distances[i])[:top_n]  # 가장 가까운 이웃 상위 n개 선택
        neighbors.append(nearest_indices)
    return neighbors

def map_indices_to_text(neighbors, original_indices, original_texts):
    '''
    원래 데이터의 인덱스를 매핑하는 함수
    '''
    index_to_text = {idx: text for idx, text in zip(original_indices, original_texts)}
    mapped_neighbors = []
    for neighbor_indices in neighbors:
        mapped_texts = [index_to_text[original_indices[idx]] for idx in neighbor_indices]
        mapped_neighbors.append(mapped_texts)
    return mapped_neighbors


# 원래 데이터의 인덱스를 텍스트로 변환하는 함수
def map_indices_to_text(neighbors, original_indices, original_texts):
    index_to_text = {idx: text for idx, text in zip(original_indices, original_texts)}
    mapped_neighbors = []
    for neighbor_indices in neighbors:
        mapped_texts = [index_to_text[original_indices[idx]] for idx in neighbor_indices]
        mapped_neighbors.append(mapped_texts)
    return mapped_neighbors

# 작업 디렉토리 설정
os.chdir('C:/Users/mose/agoda/data/')

# 데이터 로드
eng = pd.read_csv("eng.csv", index_col=0)

# 벡터화
cntvectorizer = CountVectorizer(stop_words='english')
cnt_X = cntvectorizer.fit_transform(eng['Text'])

tfidfvectorizer = TfidfVectorizer(stop_words='english')
tf_X = tfidfvectorizer.fit_transform(eng['Text'])

# # 해밍 거리 행렬 계산
# cnt_ham_distances = cal_ham_dist(cnt_X)
# tf_ham_distances = cal_ham_dist(tf_X)

# # 거리 행렬 저장
# np.save('cnt_ham_distances.npy', cnt_ham_distances)
# np.save('tf_ham_distances.npy', tf_ham_distances)

if __name__ == "__main__":
    
    os.chdir('C:/Users/mose/agoda/data/nbds-24-05-20')

    # 거리 행렬 로드
    cnt_ham_distances = np.load('cnt_ham_distances.npy')
    tf_ham_distances = np.load('tf_ham_distances.npy')

    # 대각 원소를 모두 1로 설정
    np.fill_diagonal(cnt_ham_distances, 1)
    np.fill_diagonal(tf_ham_distances, 1)

    # 상위 10개, 50개, 100개 이웃 추출
    cnt_ham_nbds_10 = nbds_sampling(cnt_ham_distances, 10)
    cnt_ham_nbds_50 = nbds_sampling(cnt_ham_distances, 50)
    cnt_ham_nbds_100 = nbds_sampling(cnt_ham_distances, 100)

    tf_ham_nbds_10 = nbds_sampling(tf_ham_distances, 10)
    tf_ham_nbds_50 = nbds_sampling(tf_ham_distances, 50)
    tf_ham_nbds_100 = nbds_sampling(tf_ham_distances, 100)

    # 결과 출력
    print("Count Vectorizer 기반 해밍 거리 상위 10개 이웃:")
    print(cnt_ham_nbds_10)

    print("Count Vectorizer 기반 해밍 거리 상위 50개 이웃:")
    print(cnt_ham_nbds_50)

    print("Count Vectorizer 기반 해밍 거리 상위 100개 이웃:")
    print(cnt_ham_nbds_100)

    print("TF-IDF Vectorizer 기반 해밍 거리 상위 10개 이웃:")
    print(tf_ham_nbds_10)

    print("TF-IDF Vectorizer 기반 해밍 거리 상위 50개 이웃:")
    print(tf_ham_nbds_50)

    print("TF-IDF Vectorizer 기반 해밍 거리 상위 100개 이웃:")
    print(tf_ham_nbds_100)

# dic 정의
o2m_dic = {orig_idx: i for i, orig_idx in enumerate(eng.index)}
m2o_dic = {i: orig_idx for i, orig_idx in enumerate(eng.index)}

