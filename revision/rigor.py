# df_6['ori-ebd']
# df_1['0.1-gen1-ebd']
# df_2['0.1-gen2-ebd']
# df_3['0.1-gen1-1-ebd']
# df_4['0.7-gen1-ebd']
# df_5['0.7-gen2-ebd']
# df_6['0.7-gen1-1-ebd']
# df_7['1.5-gen1-ebd']
# df_8['1.5-gen2-ebd']
# df_9['1.5-gen1-1-ebd']

#%%
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import ast
from tqdm import tqdm


# 문자열 데이터를 NumPy 배열로 변환하는 함수
def string_to_vector(data):
    try:
        if isinstance(data, str):
            return np.array(ast.literal_eval(data))
        elif isinstance(data, (list, np.ndarray)):
            return np.array(data)
        else:
            raise ValueError("Unsupported data type")
    except (ValueError, SyntaxError) as e:
        raise TypeError(f"Failed to convert data to vector: {data}. Error: {e}")

# 거리 함수 정의
def calculate_distances(vec1, vec2):
    cosine_sim = 1 - cosine(vec1, vec2)  # cosine
    kl_div = entropy(vec1, vec2)  # KL divergence
    wasserstein = wasserstein_distance(vec1, vec2)  # Wasserstein distance
    return {
        "cosine_similarity": cosine_sim,
        "kl_divergence": kl_div,
        "wasserstein_distance": wasserstein
    }

#%%
'''
데이터 로드
'''
# os.chdir('C:/Users/mose/agoda/data/') # home
# os.chdir('C:/Users/UOS/Desktop/Agoda-Data/') # lab
# os.chdir('/home1/mose1103/agoda/LLM') # UBAI
os.chdir('D:/mose/data/ablation2') # D-drive


df_1 = pd.read_csv("df_1.csv", index_col=0, encoding='utf-8-sig') # 0.1-gen1-ebd
df_2 = pd.read_csv("df_2.csv", index_col=0, encoding='utf-8-sig') # 0.1-gen2-ebd
df_3 = pd.read_csv("df_3.csv", index_col=0, encoding='utf-8-sig') # 0.1-gen1-1-ebd
df_4 = pd.read_csv("df_4.csv", index_col=0, encoding='utf-8-sig') # 0.7-gen1-ebd
df_5 = pd.read_csv("df_5.csv", index_col=0, encoding='utf-8-sig') # 0.7-gen2-ebd
df_6 = pd.read_csv("df_6.csv", index_col=0, encoding='utf-8-sig') # 0.7-gen1-1-ebd
df_7 = pd.read_csv("df_7.csv", index_col=0, encoding='utf-8-sig') # 1.5-gen1-ebd
df_8 = pd.read_csv("df_8.csv", index_col=0, encoding='utf-8-sig') # 1.5-gen2-ebd
df_9 = pd.read_csv("df_9.csv", index_col=0, encoding='utf-8-sig') # 1.5-gen1-1-ebd

#%%
# 결과 저장용 딕셔너리
results = {
    "0.1": {"cosine": [], "kl": [], "wasserstein": []},
    "0.7": {"cosine": [], "kl": [], "wasserstein": []},
    "1.5": {"cosine": [], "kl": [], "wasserstein": []}
}

# 각 단계별 데이터 처리
for scale, (ori_col, gen1_col, gen2_col, gen1_1_col) in {
    "0.1": (df_6['ori-ebd'], df_1['0.1-gen1-ebd'], df_2['0.1-gen2-ebd'], df_3['0.1-gen1-1-ebd']),
    "0.7": (df_6['ori-ebd'], df_4['0.7-gen1-ebd'], df_5['0.7-gen2-ebd'], df_6['0.7-gen1-1-ebd']),
    "1.5": (df_6['ori-ebd'], df_7['1.5-gen1-ebd'], df_8['1.5-gen2-ebd'], df_9['1.5-gen1-1-ebd'])
}.items():
    for i in tqdm(range(len(df_1)), desc=f"Processing {scale} distances"):
        # 각 행에서 벡터화
        ori_vec = string_to_vector(ori_col.iloc[i])
        gen1_vec = string_to_vector(gen1_col.iloc[i])
        gen2_vec = string_to_vector(gen2_col.iloc[i])
        gen1_1_vec = string_to_vector(gen1_1_col.iloc[i])

        # 거리 계산
        results[scale]["cosine"].append({
            "ori-gen1": 1 - cosine(ori_vec, gen1_vec),
            "ori-gen2": 1 - cosine(ori_vec, gen2_vec),
            "gen1-gen1-1": 1 - cosine(gen1_vec, gen1_1_vec),
            "gen2-gen1-1": 1 - cosine(gen2_vec, gen1_1_vec)
        })

        results[scale]["kl"].append({
            "ori-gen1": entropy(ori_vec, gen1_vec + 1e-10),
            "ori-gen2": entropy(ori_vec, gen2_vec + 1e-10),
            "gen1-gen1-1": entropy(gen1_vec, gen1_1_vec + 1e-10),
            "gen2-gen1-1": entropy(gen2_vec, gen1_1_vec + 1e-10)
        })

        results[scale]["wasserstein"].append({
            "ori-gen1": wasserstein_distance(ori_vec, gen1_vec),
            "ori-gen2": wasserstein_distance(ori_vec, gen2_vec),
            "gen1-gen1-1": wasserstein_distance(gen1_vec, gen1_1_vec),
            "gen2-gen1-1": wasserstein_distance(gen2_vec, gen1_1_vec)
        })

# 결과 확인
for scale in results:
    print(f"\nResults for {scale}:")
    print("Cosine Similarities:", results[scale]["cosine"][:5])
    print("KL Divergences:", results[scale]["kl"][:5])
    print("Wasserstein Distances:", results[scale]["wasserstein"][:5])
