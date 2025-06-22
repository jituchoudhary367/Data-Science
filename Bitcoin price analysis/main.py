import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"data\bitcoin.csv")
df.head()

df.shape
df.describe()
df.info()

plt.figure(figsize=(10,7))
plt.plot(df['Close'])
plt.title("Bitcoin Close Price",fontsize = 15)
plt.xlabel("Variation in Price of Cryptocurrency")
plt.ylabel("Price in dollars")
plt.show()

df[df["Close"] == df["Adj Close"]].shape,df.shape
df = df.drop(["Adj Close"],axis=1)
df.isnull().sum()

features = ["Open", "High", "Low", "Close"]
plt.subplots(figsize=(10,5))
for i , col in enumerate(features):
    plt.subplot(2,2,i+1)
    sns.displot(df[col])
plt.show()

plt.subplots(figsize=(10,5))
for i, col in enumerate(features):
  plt.subplot(2,2,i+1)
  sns.boxplot(df[col])
plt.show()

splitted = df['Date'].str.split('-', expand=True)

df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')

# Convert the 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date']) 
df.head()

data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
plt.show()

df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df.head()

df['open-close'] = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

plt.pie(df['target'].value_counts().values, 
        labels=[0, 1], autopct='%1.1f%%')
plt.show()

plt.figure(figsize=(10,10))
sns.heatmap(df.corr() > 0.9, annot=True , cbar = False)
plt.show()

features = df[['open-close','low-high','is_quarter_end']]
target = df['target']
scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = X_train, X_valid, Y_train, Y_valid = features[:len(features)//7],features[len(features)//7:],target[:len(features)//7],target[len(features)//7:]

models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid)
plt.show()