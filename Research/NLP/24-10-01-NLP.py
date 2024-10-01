'''
NLP 4가지 task

이 코드는 딥러닝을 이용한 자연어 처리 입문 (유원준, 상준) 에게 저작권이 있으며 영리적인 목적으로 코드를 사용하지 않음을 말씀드립니다.

1. 불용어 제거 : ‘for’, ‘also’, ‘can’, ‘the’
2. 도메인 특화 단어 제거 : ‘statistician’, ‘estimate’, ‘sample’
3. 스테밍 : ‘play’, ‘player’, ‘playing
4. 토크나이제이션 : 각 단어 개수의 벡터로 변환
'''

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from konlpy.tag import Okt

stop_words_list = stopwords.words('english')


#%% 

'''
1. 불용어 제거
'''

example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english')) 

word_tokens = word_tokenize(example)

# # 불용어 목록에 새로운 단어 추가
# custom_stop_words = {"family", "important"}
# stop_words.update(custom_stop_words)

result = []
for word in word_tokens: 
    if word not in stop_words: 
        result.append(word) 

print('불용어 제거 전 :',word_tokens) 
print('불용어 제거 후 :',result)

#%%

'''
스테밍
'''

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize

# 포터 알고리즘
stemmer = PorterStemmer()

sentence = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
tokenized_sentence = word_tokenize(sentence)

print('어간 추출 전 :', tokenized_sentence)
print('어간 추출 후 :',[stemmer.stem(word) for word in tokenized_sentence])



# 랭캐스터알고리즘 (비교)

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print('어간 추출 전 :', words)
print('포터 스테머의 어간 추출 후:',[porter_stemmer.stem(w) for w in words])
print('랭커스터 스테머의 어간 추출 후:',[lancaster_stemmer.stem(w) for w in words])


#%%

'''
토크나이제이션
'''

# 1. Bag of words

from sklearn.feature_extraction.text import CountVectorizer

# 예시 문장들
sentences = ["The cat sat on the mat.", "The dog lay on the rug."]

# CountVectorizer를 사용한 Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# 결과 확인
print(vectorizer.get_feature_names_out())  # 단어 목록
print(X.toarray())  # 각 문장의 벡터

# 2. TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

# 예시 문장들
sentences = ["The cat sat on the mat.", "The dog lay on the rug."]

# TfidfVectorizer를 사용한 TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

# 결과 확인
print(vectorizer.get_feature_names_out())  # 단어 목록
print(X.toarray())  # 각 문장의 벡터

# 3. 워드 임베딩 (word2vec)

from gensim.models import Word2Vec

# 예시 문장
sentences = [["The", "cat", "sat", "on", "the", "mat"], ["The", "dog", "lay", "on", "the", "rug"]]

# Word2Vec 모델 훈련
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=4)

# "cat" 단어의 벡터 확인
print(model.wv['cat'])

# 4. n-gram

from sklearn.feature_extraction.text import CountVectorizer

# 예시 문장
sentences = ["The cat sat on the mat.", "The dog lay on the rug."]

# 2-gram 사용
vectorizer = CountVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(sentences)

# 결과 확인
print(vectorizer.get_feature_names_out())  # n-gram 목록
print(X.toarray())  # 각 문장의 n-gram 벡터

# 5. Tokenization and Counting Using NLTK or SpaCy

from nltk.tokenize import word_tokenize
from collections import Counter

# 예시 문장
example = "Family is not an important thing. It's everything."

# 토큰화
word_tokens = word_tokenize(example)

# 각 단어의 빈도 계산
word_count = Counter(word_tokens)

# 결과 출력
print(word_count)
