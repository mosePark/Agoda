import pandas as pd

# data load
df = pd.read_csv('data/train.csv')

# data summarization
df.describe()

# data type
df.info()

# binary val -> 1 or 0
data['col_name'].factorize()

# one-hot encdoing
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
embarked_encoded, embarked_categories = df['col_name_2'].factorize()
embarked_hot = encoder.fit_transform(embarked_encoded.reshape(-1, 1))
print(embarked_hot.toarray())

# normalization, minmax scaling
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
fitted = min_max_scaler.fit(df)
fitted.data_max_

output = min_max_scaler.transform(df)
output = pd.DataFrame(output, columns=df.columns, index=list(df.index.values))
output.head()

# standardization
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
df.head()

fitted = std_scaler.fit(df)
fitted.mean_

output = std_scaler.transform(df)
output = pd.DataFrame(output, columns=df.columns, index=list(df.index.values))
output.head()
