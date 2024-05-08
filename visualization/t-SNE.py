from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 데이터 로드 및 샘플링
digits = load_digits()
X = digits.data[:300]  # 데이터 감소
y = digits.target[:300]

# PCA로 차원 축소
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X)

# t-SNE 모델 생성 및 데이터 변환
tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X_pca)

# t-SNE 시각화
plt.figure(figsize=(10, 8))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.colorbar()
plt.title("t-SNE Visualization of the Digits Dataset (Reduced Data)")
plt.show()
