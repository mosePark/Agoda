'''
1. clustering 진행할때 차원축소 후 진행, 클러스터링 작업 후
2. permutation testing 
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

df = pd.read_csv("ori-and-gen.csv", index_col='Unnamed: 0')

#%%

df
