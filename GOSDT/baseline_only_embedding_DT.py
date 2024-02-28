import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


os.chdir('C:/Users/UOS/proj_0/GOSDT/gosdt-guesses/baseline/')

# data load
df = pd.read_excel("base_data.xlsx", index_col=0)
ebd = pd.read_csv('embedding_100.csv', index_col=0)

# target variable
df['y-y_hat'] = np.abs(df['Score'] - df['med_y_hat'])

# only embedding
var = ['hotel_name', 'y-y_hat']
df = df.loc[:, var]

df = pd.concat([df, ebd], axis=1)
df.head()

'''
y : binary
'''
df['y-y_hat'] = np.abs(df['y-y_hat']).apply(lambda x: 1 if x >= 2 else 0)



nums = df.columns[2:].tolist()
new_order = ['hotel_name'] + nums + ['y-y_hat']

df= df[new_order]


X, y = df.iloc[:,2:], df.iloc[:,-1]

x_train, x_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True
)

model = DecisionTreeClassifier(
    max_features='sqrt', # 노드 분할에 사용될 변수 최대의 수
    max_depth=100,
    max_leaf_nodes=50,
    random_state=240228
)

model.fit(x_train, y_train)

def get_clf_eval(y_test, y_pred=None):
    confusion = confusion_matrix(y_test, y_pred, labels=[True, False])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=[True, False])
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, labels=[True, False])

    print("오차행렬:\n", confusion)
    print("\n정확도: {:.4f}".format(accuracy))
    print("정밀도: {:.4f}".format(precision))
    print("재현율: {:.4f}".format(recall))
    print("F1: {:.4f}".format(F1))

pred = model.predict(x_val)
get_clf_eval(y_val, pred)

# 트리 시각화
plt.figure(figsize=(20,10))
tree.plot_tree(model, 
               feature_names=x_train.columns,  
               class_names=[str(c) for c in model.classes_],
               filled=True,
               proportion=True,
               node_ids=True
               )  
plt.show()
