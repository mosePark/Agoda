'''
임베딩 잘못해서 수정함
- gen1만 두번 임베딩했기 때문
'''

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
#%%
'''
API 로드
'''

# API 키 정보 로드
current_directory = os.getcwd()
project_root = current_directory

# .env 파일 경로 설정
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)

# API 키를 환경 변수에서 가져오기
api_key = os.getenv("OPENAI_API_KEY")

#%%
'''
임베딩
'''

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    if isinstance(text, str):
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    else:
        return np.zeros(1536)  # 빈 텍스트나 NaN 값에 대한 기본 임베딩 벡터 (예: 1536차원 벡터)

def string_to_array(string):
    try:
        return np.array(ast.literal_eval(string))
    except (ValueError, SyntaxError) as e:
        print(f"문자열을 배열로 변환하는 중 오류 발생: {e}")
        return None

#%%

'''
데이터 로드
'''

# os.chdir('C:/Users/mose/agoda/data/') # 집
os.chdir('C:/Users/UOS/Desktop/Agoda-Data/raw') # 연구실

gen_ebd = pd.read_csv("gen1_ebd.csv", index_col='Unnamed: 0')
gen = pd.read_csv("gen_2.csv", index_col='Unnamed: 0')

#%%
'''
임베딩 생성
'''

# tqdm과 pandas 통합
tqdm.pandas()

# 진행 상황 모니터링을 위한 코드 수정

gen["gen_2_embd"] = gen["generated_review_2"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))

# gen_1.to_csv("gen1_ebd.csv", encoding="utf-8-sig")
# gen_2.to_csv("gen2_ebd.csv", encoding="utf-8-sig")

#%%
'''
KMeans 클러스터링 수행
'''

# 임베딩 리스트로 변환
ori_embeddings = np.vstack(gen_ebd["ori_embd"].apply(string_to_array).to_numpy())
gen1_embeddings = np.vstack(gen_ebd["gen_1_embd"].apply(string_to_array).to_numpy())
gen2_embeddings = np.vstack(gen["gen_2_embd"].to_numpy())

# KMeans 클러스터링 수행 (5개의 클러스터)
kmeans_ori = KMeans(n_clusters=5, random_state=240902).fit(ori_embeddings)
kmeans_gen1 = KMeans(n_clusters=5, random_state=240902).fit(gen1_embeddings)
kmeans_gen2 = KMeans(n_clusters=5, random_state=240902).fit(gen2_embeddings)

#%%

# 클러스터 라벨 추출
labels_ori = kmeans_ori.labels_
labels_gen1 = kmeans_gen1.labels_
labels_gen2 = kmeans_gen2.labels_
# %%

#%%
'''
평가
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

#%%

'''
시각화
'''

# 클러스터링 PCA 시각화

pca = PCA(n_components=2, random_state=240902)

# 원본 리뷰 임베딩을 2D 공간으로 변환
ori_2d = pca.fit_transform(ori_embeddings)
explained_variance_ori = pca.explained_variance_ratio_
print(f"원본 리뷰 PCA 설명된 분산 비율: {explained_variance_ori}")

# 생성된 리뷰 (gen1) 임베딩을 2D 공간으로 변환
gen1_2d = pca.fit_transform(gen1_embeddings)
explained_variance_gen1 = pca.explained_variance_ratio_
print(f"생성된 리뷰 (gen1) PCA 설명된 분산 비율: {explained_variance_gen1}")

# 생성된 리뷰 (gen2) 임베딩을 2D 공간으로 변환
gen2_2d = pca.fit_transform(gen2_embeddings)
explained_variance_gen2 = pca.explained_variance_ratio_
print(f"생성된 리뷰 (gen2) PCA 설명된 분산 비율: {explained_variance_gen2}")

# 결과 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes = axes.flatten()

# 클러스터 색상 정의
colors = ['red', 'green', 'blue', 'purple', 'orange']

# 원본 리뷰 클러스터링
for cluster in range(5):
    axes[0].scatter(ori_2d[labels_ori == cluster, 0], ori_2d[labels_ori == cluster, 1], 
                    c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)
axes[0].set_title('원본 리뷰 클러스터링')
axes[0].legend()

# 생성된 리뷰 (gen1) 클러스터링
for cluster in range(5):
    axes[1].scatter(gen1_2d[labels_gen1 == cluster, 0], gen1_2d[labels_gen1 == cluster, 1], 
                    c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)
axes[1].set_title('생성된 리뷰 (gen1) 클러스터링')
axes[1].legend()

# 생성된 리뷰 (gen2) 클러스터링
for cluster in range(5):
    axes[2].scatter(gen2_2d[labels_gen2 == cluster, 0], gen2_2d[labels_gen2 == cluster, 1], 
                    c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)
axes[2].set_title('생성된 리뷰 (gen2) 클러스터링')
axes[2].legend()

plt.tight_layout()
plt.show()
# %%

# 3D PCA 변환
pca_3d = PCA(n_components=3, random_state=240902)

# 원본 리뷰 임베딩을 3D 공간으로 변환
ori_3d = pca_3d.fit_transform(ori_embeddings)
explained_variance_ori_3d = pca_3d.explained_variance_ratio_
print(f"원본 리뷰 PCA 설명된 분산 비율 (3D): {explained_variance_ori_3d}")

# 생성된 리뷰 (gen1) 임베딩을 3D 공간으로 변환
gen1_3d = pca_3d.fit_transform(gen1_embeddings)
explained_variance_gen1_3d = pca_3d.explained_variance_ratio_
print(f"생성된 리뷰 (gen1) PCA 설명된 분산 비율 (3D): {explained_variance_gen1_3d}")

# 생성된 리뷰 (gen2) 임베딩을 3D 공간으로 변환
gen2_3d = pca_3d.fit_transform(gen2_embeddings)
explained_variance_gen2_3d = pca_3d.explained_variance_ratio_
print(f"생성된 리뷰 (gen2) PCA 설명된 분산 비율 (3D): {explained_variance_gen2_3d}")

# 3D 결과 시각화
fig = plt.figure(figsize=(18, 6))

# 원본 리뷰 클러스터링
ax1 = fig.add_subplot(131, projection='3d')
for cluster in range(5):
    ax1.scatter(ori_3d[labels_ori == cluster, 0], ori_3d[labels_ori == cluster, 1], ori_3d[labels_ori == cluster, 2],
                c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)
ax1.set_title('원본 리뷰 클러스터링')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')
ax1.legend()

# 생성된 리뷰 (gen1) 클러스터링
ax2 = fig.add_subplot(132, projection='3d')
for cluster in range(5):
    ax2.scatter(gen1_3d[labels_gen1 == cluster, 0], gen1_3d[labels_gen1 == cluster, 1], gen1_3d[labels_gen1 == cluster, 2],
                c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)
ax2.set_title('생성된 리뷰 (gen1) 클러스터링')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_zlabel('PC3')
ax2.legend()

# 생성된 리뷰 (gen2) 클러스터링
ax3 = fig.add_subplot(133, projection='3d')
for cluster in range(5):
    ax3.scatter(gen2_3d[labels_gen2 == cluster, 0], gen2_3d[labels_gen2 == cluster, 1], gen2_3d[labels_gen2 == cluster, 2],
                c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)
ax3.set_title('생성된 리뷰 (gen2) 클러스터링')
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_zlabel('PC3')
ax3.legend()

plt.tight_layout()
plt.show()
# %%

# 상자 그림 시각화를 위한 데이터 준비
scores_ori = [gen['Score'][labels_ori == cluster].values for cluster in range(5)]
scores_gen1 = [gen['Score'][labels_gen1 == cluster].values for cluster in range(5)]
scores_gen2 = [gen['Score'][labels_gen2 == cluster].values for cluster in range(5)]

# 각 클러스터의 점수 분포 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 원본 리뷰
axes[0].boxplot(scores_ori, patch_artist=True, notch=True)
axes[0].set_title('원본 리뷰 점수 분포')
axes[0].set_xlabel('클러스터')
axes[0].set_ylabel('점수')
axes[0].set_xticklabels([f'클러스터 {i}' for i in range(5)])

# 생성된 리뷰 (gen1)
axes[1].boxplot(scores_gen1, patch_artist=True, notch=True)
axes[1].set_title('생성된 리뷰 (gen1) 점수 분포')
axes[1].set_xlabel('클러스터')
axes[1].set_ylabel('점수')
axes[1].set_xticklabels([f'클러스터 {i}' for i in range(5)])

# 생성된 리뷰 (gen2)
axes[2].boxplot(scores_gen2, patch_artist=True, notch=True)
axes[2].set_title('생성된 리뷰 (gen2) 점수 분포')
axes[2].set_xlabel('클러스터')
axes[2].set_ylabel('점수')
axes[2].set_xticklabels([f'클러스터 {i}' for i in range(5)])

plt.tight_layout()
plt.show()
# %%
# 생성 텍스트 클러스터 2번 데이터 조회

# gen1에서 클러스터 2에 속한 리뷰 조회
cluster2_gen1_indices = np.where(labels_gen1 == 2)[0]
cluster2_gen1_reviews = gen.iloc[cluster2_gen1_indices]
print("생성된 리뷰 (gen1) - 클러스터 2:")
cluster2_gen1_reviews

# gen2에서 클러스터 2에 속한 리뷰 조회
cluster2_gen2_indices = np.where(labels_gen2 == 2)[0]
cluster2_gen2_reviews = gen.iloc[cluster2_gen2_indices]
print("\n생성된 리뷰 (gen2) - 클러스터 2:")
cluster2_gen2_reviews

# 세트로 변환하여 비교를 쉽게 함
set_gen1 = set(cluster2_gen1_indices)
set_gen2 = set(cluster2_gen2_indices)

# 교집합 찾기 (공통 요소)
common_elements = set_gen1.intersection(set_gen2)

# 공통 요소의 개수 계산
num_common_elements = len(common_elements)

# 두 세트에서 고유 요소의 총 개수 계산 (합집합)
total_elements = len(set_gen1.union(set_gen2))

# 공통 요소의 비율 계산
proportion_common = num_common_elements / total_elements if total_elements > 0 else 0

print(f"공통 요소의 개수: {num_common_elements}")
print(f"공통 요소의 비율: {proportion_common:.2f}")



cluster2_gen2_indices = cluster2_gen2_reviews.index

with open("cluster2_idx.txt", 'w') as f:
    for index in cluster2_gen2_indices:
        f.write(str(index) + '\n')
