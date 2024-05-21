'''
오리지날 데이터 인덱스 : ?
이웃 몇개 뽑을지 : ?

숫자를 입력하면 그 숫자에 해당하는 텍스트(center)와 그 이웃 포인트를 출력
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

def get_text(neighbors, m2o_dic, df):
    original_texts = []
    for idx in neighbors:
        original_idx = m2o_dic[idx]  # 변경된 인덱스를 원래 인덱스로 변환
        original_text = df.loc[original_idx, 'Text']  # 원래 인덱스를 사용하여 텍스트를 가져옴
        original_texts.append(original_text)
    return original_texts

def center_nb_text(o_num, o2m_dic, m2o_dic, df, distances, top_n=10):
    m_num = o2m_dic[o_num]

    # 센터포인트 텍스트 출력
    center_text = df.loc[o_num, 'Text']
    print(f"Center Text (Index {o_num}):\n{center_text}\n")

    # 이웃 텍스트 계산 및 출력
    neighbors = np.argsort(distances[m_num])[:top_n]  # 가장 가까운 이웃 상위 n개 선택
    nbds_texts = get_text(neighbors, m2o_dic, df)
    print("Neighbor Texts:")
    for text in nbds_texts:
        print(f"- {text}")

def main():
    # 작업 디렉토리 설정
    os.chdir('C:/Users/mose/agoda/data/')

    # 데이터 로드
    eng = pd.read_csv("eng.csv", index_col=0)

    # 작업 디렉토리 변경
    os.chdir('C:/Users/mose/agoda/data/nbds-24-05-20')

    # 거리 행렬 로드
    cnt_ham_distances = np.load('cnt_ham_distances.npy')
    tf_ham_distances = np.load('tf_ham_distances.npy')

    # 대각 원소를 모두 1로 설정
    np.fill_diagonal(cnt_ham_distances, 1)
    np.fill_diagonal(tf_ham_distances, 1)

    # dic 정의
    o2m_dic = {orig_idx: i for i, orig_idx in enumerate(eng.index)}
    m2o_dic = {i: orig_idx for i, orig_idx in enumerate(eng.index)}

    nums = int(input())
    top_n = int(input())


    # 함수 사용 예시
    center_nb_text(
        o_num=nums,
        o2m_dic=o2m_dic,
        m2o_dic=m2o_dic,
        df=eng,
        distances=cnt_ham_distances,  # CountVectorizer를 사용한 이웃 텍스트 리스트
        top_n=top_n
    )

    center_nb_text(
        o_num=nums,
        o2m_dic=o2m_dic,
        m2o_dic=m2o_dic,
        df=eng,
        distances=tf_ham_distances,  # TfidfVectorizer를 사용한 이웃 텍스트 리스트
        top_n=top_n
    )

if __name__ == "__main__":
    main()

