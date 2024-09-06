'''
1. clustering 진행할때 차원축소 후 진행, 클러스터링 작업 후
2. permutation testing 
'''

import os
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
# 문자열을 배열로 변환하는 함수
def string_to_array(string):
    try:
        return np.array(ast.literal_eval(string))
    except (ValueError, SyntaxError) as e:
        print(f"문자열을 배열로 변환하는 중 오류 발생: {e}")
        return None

# 데이터 로드
    
# os.chdir('C:/Users/mose/agoda/data/') # 집
os.chdir('C:/Users/UOS/Desktop/Agoda-Data/raw') # 연구실

df = pd.read_csv("ori-and-gen.csv", index_col='Unnamed: 0')

# 임베딩 벡터를 배열로 변환
gen_1_embd_array = df['gen_1_embd'].apply(string_to_array).dropna().tolist()
gen_2_embd_array = df['gen_2_embd'].apply(string_to_array).dropna().tolist()
ori_embd_array = df['ori_embd'].apply(string_to_array).dropna().tolist()

# 각 임베딩 배열을 하나의 행렬로 변환
gen_1_embeddings = np.vstack(gen_1_embd_array)
gen_2_embeddings = np.vstack(gen_2_embd_array)
ori_embeddings = np.vstack(ori_embd_array)

#%%

# PCA 차원 축소
n_components = 256  # 차원 축소 후 차원 수
pca = PCA(n_components=n_components)

# 각 데이터셋에 대해 PCA 수행
gen_1_embeddings_reduced = pca.fit_transform(gen_1_embeddings)
print("Gen 1 임베딩에서 설명된 분산 비율:", np.sum(pca.explained_variance_ratio_))

gen_2_embeddings_reduced = pca.fit_transform(gen_2_embeddings)
print("Gen 2 임베딩에서 설명된 분산 비율:", np.sum(pca.explained_variance_ratio_))

ori_embeddings_reduced = pca.fit_transform(ori_embeddings)
print("Original 임베딩에서 설명된 분산 비율:", np.sum(pca.explained_variance_ratio_))

# 약 80%대 정도 분산 비율 보존


#%%
'''
hierarchical 클러스터링 수행
'''

# Hierarchical Clustering 수행 (5개의 클러스터로 설정)
agg_ori = AgglomerativeClustering(n_clusters=5).fit(ori_embeddings_reduced)
agg_gen1 = AgglomerativeClustering(n_clusters=5).fit(gen_1_embeddings_reduced)
agg_gen2 = AgglomerativeClustering(n_clusters=5).fit(gen_2_embeddings_reduced)

# 클러스터 라벨 추출
labels_ori = agg_ori.labels_
labels_gen1 = agg_gen1.labels_
labels_gen2 = agg_gen2.labels_

#%%
'''
클러스터링 평가
'''

metrics = {
    "Rand Index": [
        rand_score(labels_ori, labels_gen1),
        rand_score(labels_ori, labels_gen2),
        rand_score(labels_gen1, labels_gen2)
    ],
    "Adjusted Rand Index (ARI)": [
        adjusted_rand_score(labels_ori, labels_gen1),
        adjusted_rand_score(labels_ori, labels_gen2),
        adjusted_rand_score(labels_gen1, labels_gen2)
    ],
    "Mutual Information (MI)": [
        mutual_info_score(labels_ori, labels_gen1),
        mutual_info_score(labels_ori, labels_gen2),
        mutual_info_score(labels_gen1, labels_gen2)
    ],
    "Normalized Mutual Information (NMI)": [
        normalized_mutual_info_score(labels_ori, labels_gen1),
        normalized_mutual_info_score(labels_ori, labels_gen2),
        normalized_mutual_info_score(labels_gen1, labels_gen2)
    ],
    "Homogeneity": [
        homogeneity_score(labels_ori, labels_gen1),
        homogeneity_score(labels_ori, labels_gen2),
        homogeneity_score(labels_gen1, labels_gen2)
    ],
    "Completeness": [
        completeness_score(labels_ori, labels_gen1),
        completeness_score(labels_ori, labels_gen2),
        completeness_score(labels_gen1, labels_gen2)
    ],
    "V-Measure": [
        v_measure_score(labels_ori, labels_gen1),
        v_measure_score(labels_ori, labels_gen2),
        v_measure_score(labels_gen1, labels_gen2)
    ]
}

# 보기 쉽게 DataFrame 생성
comparison_pairs = [
    "Original vs Gen1",
    "Original vs Gen2",
    "Gen1 vs Gen2"
]

results = pd.DataFrame(metrics, index=comparison_pairs)
results

# %%
'''
시각화
'''

# Compute the linkage matrix for original embeddings
linked = linkage(ori_embeddings_reduced, 'ward')

# Plot
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True, no_labels=True)
plt.title("Dendrogram for Original Embeddings")
plt.show()
# %%

# testing

# 각 클러스터의 원래 레이블을 섞기 위한 함수
def permute_labels(labels):
    
    permuted_labels = np.random.permutation(labels)  # 전체 레이블을 무작위로 섞음
    return permuted_labels


# 시뮬레이션 횟수
num_simulations = 1000

# 평가 지표를 저장할 리스트 초기화
rand_index_dist = []
ari_dist = []
mi_dist = []
nmi_dist = []
homogeneity_dist = []
completeness_dist = []
v_measure_dist = []

# 500회 시뮬레이션을 수행하여 분포를 생성
for _ in tqdm(range(num_simulations)):
    np.random.seed(None)
    # 각 클러스터링 결과에 대해 레이블을 섞음
    permuted_ori = permute_labels(labels_ori)
    permuted_gen1 = permute_labels(labels_gen1)
    permuted_gen2 = permute_labels(labels_gen2)
    
    # 평가 지표 계산 및 저장
    rand_index_dist.append([
        rand_score(permuted_ori, permuted_gen1),
        rand_score(permuted_ori, permuted_gen2),
        rand_score(permuted_gen1, permuted_gen2)
    ])
    ari_dist.append([
        adjusted_rand_score(permuted_ori, permuted_gen1),
        adjusted_rand_score(permuted_ori, permuted_gen2),
        adjusted_rand_score(permuted_gen1, permuted_gen2)
    ])
    mi_dist.append([
        mutual_info_score(permuted_ori, permuted_gen1),
        mutual_info_score(permuted_ori, permuted_gen2),
        mutual_info_score(permuted_gen1, permuted_gen2)
    ])
    nmi_dist.append([
        normalized_mutual_info_score(permuted_ori, permuted_gen1),
        normalized_mutual_info_score(permuted_ori, permuted_gen2),
        normalized_mutual_info_score(permuted_gen1, permuted_gen2)
    ])
    homogeneity_dist.append([
        homogeneity_score(permuted_ori, permuted_gen1),
        homogeneity_score(permuted_ori, permuted_gen2),
        homogeneity_score(permuted_gen1, permuted_gen2)
    ])
    completeness_dist.append([
        completeness_score(permuted_ori, permuted_gen1),
        completeness_score(permuted_ori, permuted_gen2),
        completeness_score(permuted_gen1, permuted_gen2)
    ])
    v_measure_dist.append([
        v_measure_score(permuted_ori, permuted_gen1),
        v_measure_score(permuted_ori, permuted_gen2),
        v_measure_score(permuted_gen1, permuted_gen2)
    ])

# 각 지표의 분포를 배열로 변환
rand_index_dist = np.array(rand_index_dist)
ari_dist = np.array(ari_dist)
mi_dist = np.array(mi_dist)
nmi_dist = np.array(nmi_dist)
homogeneity_dist = np.array(homogeneity_dist)
completeness_dist = np.array(completeness_dist)
v_measure_dist = np.array(v_measure_dist)

# 메트릭 이름과 분포 배열 매핑
metric_to_dist = {
    "Rand Index": rand_index_dist,
    "Adjusted Rand Index (ARI)": ari_dist,
    "Mutual Information (MI)": mi_dist,
    "Normalized Mutual Information (NMI)": nmi_dist,
    "Homogeneity": homogeneity_dist,
    "Completeness": completeness_dist,
    "V-Measure": v_measure_dist
}

# 원래 평가 지표가 분포에서 어느 quantile에 위치하는지 계산
quantiles = {
    metric: [
        np.mean(metric_to_dist[metric][:, i] <= results.loc[comparison_pairs[i], metric])
        for i in range(3)
    ]
    for metric in metric_to_dist
}

quantiles_df = pd.DataFrame(quantiles, index=comparison_pairs)

print("\n원래 평가 지표의 quantiles:")
print(quantiles_df)

# 각 지표에 대한 히스토그램을 그리기
for metric, dist in zip(metrics.keys(), [rand_index_dist, ari_dist, mi_dist, nmi_dist, homogeneity_dist, completeness_dist, v_measure_dist]):
    plt.figure(figsize=(10, 6))
    
    # 각 비교 쌍에 대해 개별 히스토그램 그리기
    for i, comparison_pair in enumerate(comparison_pairs):
        plt.hist(dist[:, i], bins=30, alpha=0.7, label=comparison_pair)
    
    # 원래 결과의 평균 값 표시
    plt.axvline(x=results[metric].mean(), color='r', linestyle='--', linewidth=2, label='Original Mean')
    
    plt.title(f'{metric} Distribution in Permutation Testing')
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
