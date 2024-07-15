'''

- 각 실제점수대별의 실제값과 예측값의 차이 그래프 그려보기
x : 실제점수 , y : 실제-예측 

  * 4~5점 GPT가 잘 미스가 되있는지? 아니면 잘 맞춘것도 잘 있는지? 확인해보고자. 
  * 4개 다 그려보기. 

'''



# %%
import os
import re
from matplotlib import font_manager, rc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# 모든 폰트 경로를 가져옴
font_paths = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

# NanumGothic 폰트 경로를 찾음
nanumgothic_paths = [path for path in font_paths if 'NanumGothic' in path]

# 경로 출력
for path in nanumgothic_paths:
    print(path)

# %%
# 한글 폰트 설정
font_path = 'C:/Windows/Fonts/NanumGothic.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# %%
def extract_number(text):
    match = re.search(r'\d+\.\d+', text)
    return float(match.group()) if match else None

# %%
os.chdir('C:/Users/mose/Desktop')

df1 = pd.read_csv("리뷰.csv")
df2 = pd.read_csv("국가+리뷰.csv")
df3 = pd.read_csv("여행객+리뷰.csv")
df4 = pd.read_csv("국가+리뷰+여행객.csv")
# %%
# 빈 데이터프레임 생성
columns = ['hotel_name', 'Score', 'Country', 'Traveler Type', 'Room Type',
           'Stay Duration', 'Title', 'Text', 'Date']
df = pd.DataFrame(columns=columns)

# %%
# df4의 컬럼을 df로 복사
for column in columns:
    df[column] = df4[column]

# %%
df['y_hat'] = df1['y_hat'].apply(extract_number)
df['y_hat_'] = df2['y_hat__'].apply(extract_number)
df['y_hat__'] = df3['y_hat__'].apply(extract_number)
df['y_hat___'] = df4['y_hat__'].apply(extract_number)

# %%
df.head()

# %%
df.isnull().sum()

# %%

# y_hat : 리뷰만
# y_hat_ : 국가, 리뷰
# y_hat__ : 여행객, 리뷰
# y_hat___ : 국가, 여행객, 리뷰

# Creating the plots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1
axs[0, 0].scatter(df['Score'], df['Score'] - df['y_hat'])
axs[0, 0].set_title('Score vs. (Score - y_hat)')
axs[0, 0].set_xlabel('실제점수')
axs[0, 0].set_ylabel('실제점수 - 예측점수 (y_hat)')

# Plot 2
axs[0, 1].scatter(df['Score'], df['Score'] - df['y_hat_'])
axs[0, 1].set_title('Score vs. (Score - y_hat_)')
axs[0, 1].set_xlabel('실제점수')
axs[0, 1].set_ylabel('실제점수 - 예측점수 (y_hat_)')

# Plot 3
axs[1, 0].scatter(df['Score'], df['Score'] - df['y_hat__'])
axs[1, 0].set_title('Score vs. (Score - y_hat__)')
axs[1, 0].set_xlabel('실제점수')
axs[1, 0].set_ylabel('실제점수 - 예측점수 (y_hat__)')

# Plot 4
axs[1, 1].scatter(df['Score'], df['Score'] - df['y_hat___'])
axs[1, 1].set_title('Score vs. (Score - y_hat___)')
axs[1, 1].set_xlabel('실제점수')
axs[1, 1].set_ylabel('실제점수 - 예측점수 (y_hat___)')

plt.tight_layout()
plt.show()

# %%
# 그래프 생성
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 각 그래프에 대해 반복
for ax, y_hat in zip(axs.flat, ['y_hat', 'y_hat_', 'y_hat__', 'y_hat___']):
    x = df['Score'].values.reshape(-1, 1)
    y = np.abs(df['Score'] - df[y_hat]).values.reshape(-1, 1)
    
    # 산점도
    ax.scatter(x, y, label='Data Points')
    
    # 그래프 제목 및 라벨 설정
    ax.set_title(f'Score vs. abs(Score - {y_hat})')
    ax.set_xlabel('실제점수')
    ax.set_ylabel(f'abs(실제점수 - 예측점수 ({y_hat}))')
    ax.legend()

plt.tight_layout()
plt.show()
# %%
# 히스토그램 생성
plt.figure(figsize=(10, 6))
plt.hist(df['Score'], bins=range(1, 11), edgecolor='black', alpha=0.7)
plt.title('Distribution of Actual Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.xticks(range(1, 11))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
# %%

# 점수대 나눠서 MSE 계산
# 점수대를 나누기 위한 함수

def get_score_category(score):
    return int(score)  # 1점대, 2점대, ..., 9점대

# 점수대를 컬럼으로 추가
df['ScoreCategory'] = df['Score'].apply(get_score_category)

# 예측 점수 컬럼 목록
prediction_columns = ['y_hat', 'y_hat_', 'y_hat__', 'y_hat___']

# 각 점수대별로 MSE 계산
mse_results = {col: [] for col in prediction_columns}
categories = range(1, 10)

for score_category in categories:
    category_df = df[df['ScoreCategory'] == score_category]
    for col in prediction_columns:
        if not category_df.empty:
            mse = np.mean((category_df['Score'] - category_df[col])**2)
            mse_results[col].append(mse)
        else:
            mse_results[col].append(np.nan)  # 데이터가 없는 경우 NaN으로 채움

# %%
# 꺾은선 그래프 생성
plt.figure(figsize=(12, 6))

for col in prediction_columns:
    plt.plot(categories, mse_results[col], marker='o', label=col)

plt.title('MSE by Score Category')
plt.xlabel('Score Category')
plt.ylabel('MSE')
plt.xticks(categories)
plt.legend()
plt.grid(True)
plt.show()
# %%
