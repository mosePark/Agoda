import pandas as pd
import re
import os

os.chdir('C:/Users/UOS/proj_0/Agoda/GPT-Application/')

df = pd.read_csv("여행객유형+리뷰 예측 최종.csv")

def extract_number(text):
    match = re.search(r'\d+\.\d+', text)
    return float(match.group()) if match else None

# Apply the function to the 'y_hat__' column
df['extracted_number'] = df['y_hat__'].apply(extract_number)

df.head()
