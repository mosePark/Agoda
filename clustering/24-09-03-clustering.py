#%%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
#%%
'''
API 로드
'''

# API KEY 정보로드
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
        return client.embeddings.create(input = [text], model=model).data[0].embedding
    else:
        return np.zeros(1536)  # 빈 텍스트나 NaN 값에 대한 기본 임베딩 벡터 (예: 1536차원 벡터)


#%%

'''
데이터 로드
'''

os.chdir('C:/Users/mose/agoda/data/') # home
# os.chdir('C:/Users/UOS/Desktop/Agoda-Data/raw') # lab


gen_1 = pd.read_csv("gen_1.csv", index_col='Unnamed: 0')
gen_2 = pd.read_csv("gen_2.csv", index_col='Unnamed: 0')

np.sum(gen_1.index == gen_2.index)


#%%
'''
임베딩 생성
'''

# tqdm과 pandas 통합
tqdm.pandas()

# 진행 상황 모니터링을 위한 코드 수정
gen_1["ori_embd"] = gen_1["Full"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))
gen_1["gen_1_embd"] = gen_1["generated_review"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))
gen_2["gen_1_embd"] = gen_2["generated_review"].progress_apply(lambda x: get_embedding(x, model="text-embedding-3-small"))

# gen_1.to_csv("gen1_ebd.csv", encoding="utf-8-sig")
# gen_2.to_csv("gen2_ebd.csv", encoding="utf-8-sig")

#%%

'''
KMeans 클러스터링 수행
'''

# 임베딩 리스트로 변환
ori_embeddings = np.vstack(gen_1["ori_embd"].to_numpy())
gen1_embeddings = np.vstack(gen_1["gen_1_embd"].to_numpy())
gen2_embeddings = np.vstack(gen_2["gen_1_embd"].to_numpy())

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
Adjusted Rand Index (ARI) 계산
'''

# 원래 리뷰와 첫 번째 생성된 리뷰 간의 클러스터링 유사성 평가
ari_ori_gen1 = adjusted_rand_score(labels_ori, labels_gen1)
print(f"Adjusted Rand Index between original and generated reviews (gen1): {ari_ori_gen1}")

# 원래 리뷰와 두 번째 생성된 리뷰 간의 클러스터링 유사성 평가
ari_ori_gen2 = adjusted_rand_score(labels_ori, labels_gen2)
print(f"Adjusted Rand Index between original and generated reviews (gen2): {ari_ori_gen2}")

# 첫 번째 생성된 리뷰와 두 번째 생성된 리뷰 간의 클러스터링 유사성 평가
ari_gen1_gen2 = adjusted_rand_score(labels_gen1, labels_gen2)
print(f"Adjusted Rand Index between gen1 and gen2 generated reviews: {ari_gen1_gen2}")


#%% 수정사항 - 

'''
임베딩 잘못해서 수정함
- gen1만 두번 임베딩했기 때문
'''

#%%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
#%%
'''
API 로드
'''

# API KEY 정보로드
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
        return client.embeddings.create(input = [text], model=model).data[0].embedding
    else:
        return np.zeros(1536)  # 빈 텍스트나 NaN 값에 대한 기본 임베딩 벡터 (예: 1536차원 벡터)


#%%

'''
데이터 로드
'''

# os.chdir('C:/Users/mose/agoda/data/') # home
os.chdir('C:/Users/UOS/Desktop/Agoda-Data/raw') # lab


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
ori_embeddings = np.vstack(gen_ebd["ori_embd"].to_numpy())
gen1_embeddings = np.vstack(gen_ebd["gen_1_embd"].to_numpy())
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
Adjusted Rand Index (ARI) 계산
'''

# 원래 리뷰와 첫 번째 생성된 리뷰 간의 클러스터링 유사성 평가
ari_ori_gen1 = adjusted_rand_score(labels_ori, labels_gen1)
print(f"Adjusted Rand Index between original and generated reviews (gen1): {ari_ori_gen1}")

# 원래 리뷰와 두 번째 생성된 리뷰 간의 클러스터링 유사성 평가
ari_ori_gen2 = adjusted_rand_score(labels_ori, labels_gen2)
print(f"Adjusted Rand Index between original and generated reviews (gen2): {ari_ori_gen2}")

# 첫 번째 생성된 리뷰와 두 번째 생성된 리뷰 간의 클러스터링 유사성 평가
ari_gen1_gen2 = adjusted_rand_score(labels_gen1, labels_gen2)
print(f"Adjusted Rand Index between gen1 and gen2 generated reviews: {ari_gen1_gen2}")
