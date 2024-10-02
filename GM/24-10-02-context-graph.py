import numpy as np
import networkx as nx
from scipy.linalg import eigh

# 그래프 생성 함수
def create_semantic_graph(words):
    G = nx.Graph()
    G.add_nodes_from(words)
    # 간단히 모든 단어를 연결 (실제로는 의미 유사도 기반 연결)
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            # 임의의 가중치 부여 (실제로는 의미 유사도로 가중치 설정)
            G.add_edge(words[i], words[j], weight=1)
    return G

# 그래프 스펙트럼 계산 함수
def graph_spectrum(G):
    L = nx.laplacian_matrix(G).todense()
    eigenvalues = np.linalg.eigvals(L)
    eigenvalues = np.sort(np.real(eigenvalues))
    return eigenvalues

# 스펙트럼 거리 계산
def spectrum_distance(spectrum1, spectrum2):
    # 두 스펙트럼의 길이를 맞춤
    min_len = min(len(spectrum1), len(spectrum2))
    spectrum1 = spectrum1[:min_len]
    spectrum2 = spectrum2[:min_len]
    return np.linalg.norm(spectrum1 - spectrum2)


# Great location and great value for money!! Very friendly staff too!!
# Amazing location and excellent value for the price! The staff was incredibly welcoming and helpful throughout our stay.

words_A = ['Great', 'location', 'and', 'great', 'value', 'for', 'money', 'Very', 'friendly', 'staff', 'too']
words_B = ['Amazing', 'location', 'and', 'excellent', 'value', 'for', 'the', 'price!', 'The', 'staff', 'was', 'incredibly', 'welcoming', 'and', 'helpful', 'throughout', 'our', 'stay']

# 그래프 생성
G_A = create_semantic_graph(words_A)
G_B = create_semantic_graph(words_B)

# 스펙트럼 계산
spectrum_A = graph_spectrum(G_A)
spectrum_B = graph_spectrum(G_B)

distance = spectrum_distance(spectrum_A, spectrum_B)
print("스펙트럼 거리:", distance) # 유사도 경우 0.74
