'''
임베딩 벡터 : tf-idf, count 벡터는 여러번 생성했을때
벡터값이 동일한지?
'''

import os

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


os.chdir('/.../data/')

# 데이터 로드
eng = pd.read_csv("eng.csv", index_col=0)

# 여러 번 임베딩 수행을 위한 함수 정의
def multiple_embeddings(vectorizer, text_data, n_times=5):
    embeddings = []
    for _ in range(n_times):
        matrix = vectorizer.fit_transform(text_data)
        embeddings.append(matrix.toarray())
    return embeddings

# 임베딩 결과 비교 함수 정의
def compare_embeddings(embeddings):
    n = len(embeddings)
    differences = []
    for i in range(n):
        for j in range(i + 1, n):
            if not np.array_equal(embeddings[i], embeddings[j]):
                diff_indices = np.where(embeddings[i] != embeddings[j])
                differences.append((i, j, diff_indices))
    return differences

def print_differences(differences, embeddings):
    if not differences:
        print("모든 임베딩 결과가 동일합니다.")
    else:
        for (i, j, diff_indices) in differences:
            print(f"임베딩 {i}와 임베딩 {j}가 다릅니다.")
            for k in range(len(diff_indices[0])):
                print(f"  - 위치 ({diff_indices[0][k]}, {diff_indices[1][k]})에서 차이가 있습니다: 값 {embeddings[i][diff_indices[0][k], diff_indices[1][k]]} (임베딩 {i}) vs 값 {embeddings[j][diff_indices[0][k], diff_indices[1][k]]} (임베딩 {j})")

# 임베딩 수행
count_vectorizer = CountVectorizer(stop_words='english')
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

count_embeddings = multiple_embeddings(count_vectorizer, eng['Text'])
tfidf_embeddings = multiple_embeddings(tfidf_vectorizer, eng['Text'])

# Count 임베딩 비교 및 결과 출력
count_differences = compare_embeddings(count_embeddings)
print("Count 임베딩 비교 결과:")
print_differences(count_differences, count_embeddings)

# TF-IDF 임베딩 비교 및 결과 출력
tfidf_differences = compare_embeddings(tfidf_embeddings)
print("TF-IDF 임베딩 비교 결과:")
print_differences(tfidf_differences, tfidf_embeddings)

'''
# 코드 예시 설명
'''

a = np.array([
    [1, 0, 2],
    [0, 1, 1],
    [1, 1, 0]
])

b = np.array([
    [1, 0, 2],
    [0, 1, 0],
    [1, 3, 0]
])

c = np.array([
    [1, 3, 2],
    [0, 1, 0],
    [4, 3, 0]
])


embeddings = [a, b, c]

# 기존 함수 사용하여 비교
differences = compare_embeddings(embeddings)

# 차이점 출력
print_differences(differences, embeddings)
