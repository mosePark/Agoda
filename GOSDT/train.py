'''
30,000여개 리뷰 데이터 전처리 후 모델링(GOSDT) 시작
특이사항은 room type text의 경우 one hot vectorization 진행
'''

import numpy as np
import pandas as pd

import os
import re
import time
import pathlib

from sklearn.ensemble import GradientBoostingClassifier

from gosdt import GOSDT
from gosdt.model.threshold_guess import compute_thresholds

os.chdir('/root/default/agoda/train/')

train = pd.read_csv("train.csv", index_col=0)

train['y-y_hat'] = np.abs(train['Score'] - train['y_hat_'])

# 모델링에 사용할 변수 선정
h = ['Traveler Type',
       'Loc_score', 'Clean_score', 'Serv_score', 'Fac_score', 'VfM_score',
       'Duration', 'Stay_year', 'stay_month',
       'Review_year', 'Review_day', 'review_month',
       'deluxe', 'king', 'queen', 'double', 'guest', 'standard', 'studio',
       'suite', 'twin', 'triple', 'family', 'smoking', 'view', 'y-y_hat'
    ]


train['Traveler Type'] = train['Traveler Type'].astype('category')
train['Stay_year'] = train['Stay_year'].astype('category')
train['stay_month'] = train['stay_month'].astype('category')
train['Review_year'] = train['Review_year'].astype('category')
train['review_month'] = train['review_month'].astype('category')
train['Review_day'] = train['Review_day'].astype('category')
train['deluxe'] = train['deluxe'].astype('category')
train['king'] = train['king'].astype('category')
train['queen'] = train['queen'].astype('category')
train['double'] = train['double'].astype('category')
train['guest'] = train['guest'].astype('category')
train['standard'] = train['standard'].astype('category')
train['studio'] = train['studio'].astype('category')
train['suite'] = train['suite'].astype('category')
train['twin'] = train['twin'].astype('category')
train['triple'] = train['triple'].astype('category')
train['family'] = train['family'].astype('category')
train['smoking'] = train['smoking'].astype('category')
train['view'] = train['view'].astype('category')

df = train.loc[:, h]


'''
y : binary
'''

df['y-y_hat'] = np.abs(df['y-y_hat']).apply(lambda x: 1 if x >= 2 else 0)

X, y = df.iloc[:,:-1].values, df.iloc[:, -1].values

'''
Modeling
'''
# GBDT parameters for threshold and lower bound guesses
n_est = 50
max_depth = 3

# guess thresholds
X = pd.DataFrame(X, columns=h[:-1])
print("X:", X.shape)
print("y:",y.shape)
X_train, thresholds, header, threshold_guess_time = compute_thresholds(X, y, n_est, max_depth)
y_train = pd.DataFrame(y)

# import pickle

# # 모든 변수를 하나의 사전(dict)에 저장
# data = {
#     "X_train": X_train,
#     "y_train": y_train,
#     "thresholds": thresholds,
#     "header": header,
#     "threshold_guess_time": threshold_guess_time
# }

# # 'wb'는 바이너리 쓰기 모드를 의미
# with open('data.pkl', 'wb') as file:
#     pickle.dump(data, file)

# X_train.to_csv("X_train.csv", encoding='utf-8-sig', index=False)
# y_train.to_csv("y_train.csv", encoding='utf-8-sig', index=False)



# guess lower bound
start_time = time.perf_counter()
clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train.values.flatten())
warm_labels = clf.predict(X_train)
elapsed_time = time.perf_counter() - start_time
lb_time = elapsed_time

# save the labels from lower bound guesses as a tmp file and return the path to it.
labelsdir = pathlib.Path('/tmp/warm_lb_labels')
labelsdir.mkdir(exist_ok=True, parents=True)
labelpath = labelsdir / 'warm_label.tmp'
labelpath = str(labelpath)
pd.DataFrame(warm_labels, columns=["class_labels"]).to_csv(labelpath, header="class_labels",index=None)


# train GOSDT model
config = {
            "regularization": 3.33e-05,
            "depth_budget": 10,
            "warm_LB": True,
            "path_to_labels": labelpath,
            "time_limit": 60,
            "similar_support": False
        }

model = GOSDT(config)

model.fit(X_train, y_train)

print("evaluate the model, extracting tree and scores", flush=True)

# get the results
train_acc = model.score(X_train, y_train)
n_leaves = model.leaves()
n_nodes = model.nodes()
time = model.utime

print("Model training time: {}".format(time))
print("Training accuracy: {}".format(train_acc))
print("# of leaves: {}".format(n_leaves))
print(model.tree)
