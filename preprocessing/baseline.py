'''
랜덤으로 샘플링한 호텔 400여개 데이터 fitting
단, 잘 정제되어있는 변수로 fitting
'''

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import openpyxl
import re
import time
import pathlib

from sklearn.ensemble import GradientBoostingClassifier

from gosdt import GOSDT
from gosdt.model.threshold_guess import compute_thresholds

# read the dataset
df = pd.read_excel("C:/Users/UOS/proj_0/GOSDT/gosdt-guesses/baseline/base.xlsx", index_col=0)

df['y-y_hat'] = df['Score'] - df['med_y_hat']

h = ['Loc_score', 'Clean_score', 'Serv_score', 'Fac_score', 'VfM_score', 'Rating', 'y-y_hat']


df = df.loc[:, h]
df = df.dropna()

X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values
h = df.columns[:-1] # 독립변수 이름들


# binary화
'''
y를 연속형에서 이진형으로 바꾸기 위한 빌드업
'''
numbers = np.abs(y)
n, bins, patches = plt.hist(numbers, bins=5, edgecolor='black')
total = len(numbers)
for i in range(len(patches)):
    height = patches[i].get_height()
    percentage = '{:.1f}%'.format(100 * (height / total))
    plt.text(patches[i].get_x() + patches[i].get_width() / 2, height, 
             percentage, ha='center', va='bottom')
plt.title('Numbers Histogram with Percentages')
plt.xlabel('Number')
plt.ylabel('Frequency')
plt.show()

df['result'] = np.abs(df['y-y_hat']).apply(lambda x: 1 if x >= 3 else 0)


y = df.iloc[:, -1].values


# GBDT parameters for threshold and lower bound guesses
n_est = 40
max_depth = 1

# guess thresholds
X = pd.DataFrame(X, columns=h)
print("X:", X.shape)
print("y:",y.shape)
X_train, thresholds, header, threshold_guess_time = compute_thresholds(X, y, n_est, max_depth) # 여기 코드 확인
y_train = pd.DataFrame(y)

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
            "regularization": 0.001,
            "depth_budget": 5,
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
