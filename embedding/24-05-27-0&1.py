'''
24-05-27
label=1 vs label=0 일때 비교
'''

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import hamming

 # 작업 디렉토리 설정
os.chdir('C:/Users/UOS/Desktop/Agoda-Data/raw')

# 데이터 로드
eng = pd.read_csv("eng.csv", index_col=0)
eng = eng.reset_index(drop=True)


# 작업 디렉토리 변경
os.chdir('C:/Users/UOS/Desktop/Agoda-Data/embedding')

# 거리 행렬 로드
cnt_ham_dist = np.load('cnt_ham_distances.npy')
tf_l2_dist = np.load('tf_l2_distances.npy')

# 대각 원소를 모두 1로 설정
np.fill_diagonal(cnt_ham_dist, np.inf)
np.fill_diagonal(tf_l2_dist, np.inf)

# 작업 디렉토리 변경
os.chdir('C:/Users/UOS/proj_0/neighborhood/')

# 특정 텍스트의 유사한 텍스트 찾기
def find_neighbors(idx, dist_matrix, top_n=5):
    dist_row = dist_matrix[idx]
    neighbors = dist_row.argsort()[:top_n]
    return [(i, eng.loc[i]['difference']) for i in neighbors]

# difference가 1인 리뷰에 대한 이웃 찾기
diff_1_idx = eng[eng['difference'] == 1].index
diff_1_neighbors_cnt = {i: find_neighbors(i, cnt_ham_dist) for i in diff_1_idx}
diff_1_neighbors_tf = {i: find_neighbors(i, tf_l2_dist) for i in diff_1_idx}

# difference가 0인 리뷰에 대한 이웃 찾기
diff_0_idx = eng[eng['difference'] == 0].index
diff_0_neighbors_cnt = {i: find_neighbors(i, cnt_ham_dist) for i in diff_0_idx}
diff_0_neighbors_tf = {i: find_neighbors(i, tf_l2_dist) for i in diff_0_idx}

# 결과 출력 및 파일 저장
def save_to_file(filename, neighbors_dict, label):
    with open(filename, 'a') as f:
        f.write(f"Difference {label} neighbors' differences:\n")
        for i, neighbors in neighbors_dict.items():
            f.write(f"Review index {i}: Neighbors: {neighbors}\n")
        f.write("\n")

output_file = 'neighbors_output.txt'

# 파일 초기화
with open(output_file, 'w') as f:
    f.write("Neighbors analysis results\n\n")

# 결과 저장
save_to_file(output_file, diff_1_neighbors_cnt, 1)
save_to_file(output_file, diff_0_neighbors_cnt, 0)
save_to_file(output_file, diff_1_neighbors_tf, 1)
save_to_file(output_file, diff_0_neighbors_tf, 0)

print(f"Results saved to {output_file}")
