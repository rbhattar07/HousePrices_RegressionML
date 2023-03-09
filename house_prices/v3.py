import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.metrics import max_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures

x=joblib.load('dict1.joblib')
y=joblib.load('dict2.joblib')

df = pd.read_csv('test.csv')

#y
imp_cols= y['imp_cols']
numeric_cols = y['numeric_cols']
categorical_cols = y['categorical_cols']
encoded_cols = y['encoded_cols']
scaler = y['scaler']
encoder = y['encoder']

#x
lr =x['lr']
ridge = x['ridge']
lasso = x['lasso']
en = x['en']
rfr= x['rfr']
svr = x['svm']
xgb = x['xgb']

df = df[imp_cols]

# Scaler
df[numeric_cols]= scaler.transform(df[numeric_cols])
# Encoder
df[encoded_cols]= encoder.transform(df[categorical_cols])
df.drop(columns=categorical_cols, inplace=True)
print(df.isnull().sum())

# Imputer
imputer = SimpleImputer(strategy='mean').fit(df[numeric_cols])
df[numeric_cols]=imputer.transform(df[numeric_cols])

# Machine Learning Models
lr = lr.predict(df)
ridge = ridge.predict(df)
lasso = lasso.predict(df)
en = en.predict(df)
rfr = rfr.predict(df)
svr = svr.predict(df)
xgb = xgb.predict(df)