#%%

import numpy as np

# 임베딩 벡터가 저장된 예시 배열 (n x d)
embeddings = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
])

# 평균 중심화
mean_centered = embeddings - np.mean(embeddings, axis=0)

# 공분산 행렬 계산
cov_matrix = np.cov(mean_centered, rowvar=False)

print("공분산 행렬:\n", cov_matrix)

#%%

from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork

# 데이터 프레임 준비
df = pd.DataFrame(data)

# 힐 클라임 검색 알고리즘을 사용하여 네트워크 구조 학습
hc = HillClimbSearch(df, scoring_method=BicScore(df))
best_model = hc.estimate()

# 베이지안 네트워크 모델 생성
model = BayesianNetwork(best_model.edges())
