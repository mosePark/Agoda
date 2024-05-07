import numpy as np
import pandas as pd

import os



os.chdir('.../raw')

eng = pd.read_csv("eng.csv", index_col=0)

eng['difference'].value_counts()

# Count vec
'''
임베딩 벡터 차원은 16528, 10245
행은 데이터의 수 그대로고
열은 유니크한 단어 수

'''

from sklearn.feature_extraction.text import CountVectorizer

cntvectorizer = CountVectorizer(
    stop_words='english'
)

cnt_X = cntvectorizer.fit_transform(eng['Text'])

cntvectorizer.get_feature_names_out()

# tf-idf vec
'''
임베딩 벡터 차원은 16528, 10245
행은 데이터의 수 그대로고
열은 유니크한 단어 수
'''
from sklearn.feature_extraction.text import TfidfVectorizer

tfidfvectorizer = TfidfVectorizer(
    stop_words='english'
)

tf_X = tfidfvectorizer.fit_transform(eng['Text'])

tfidfvectorizer.get_feature_names_out()

# hashing vec
'''
16528, 1048576
'''
from sklearn.feature_extraction.text import HashingVectorizer

hashingvectorizer = HashingVectorizer()

hash_X = hashingvectorizer.fit_transform(eng['Text'])

print(cnt_X.shape)
print(tf_X.shape)
print(hash_X.shape)
