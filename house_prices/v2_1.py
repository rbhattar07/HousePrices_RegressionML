import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.metrics import max_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

df = pd.read_csv('test.csv')
df.drop(columns=['MiscFeature', 'Fence', 'PoolQC','FireplaceQu', 'Alley', 'Id'], inplace=True)

numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes('object').columns.tolist()

# Imputing Missing values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(max_iter=10, random_state=0).fit(df[numerical_cols])
df[numerical_cols] = imputer.transform(df[numerical_cols])

# Scaling Numeric Features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(df[numerical_cols])
df[numerical_cols] = scaler.transform(df[numerical_cols])

# Encoding Categorical columns
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
df[encoded_cols] = encoder.transform(df[categorical_cols])

df1 = df.copy()

df1 = pd.concat([df1[numerical_cols], df1[encoded_cols]], axis=1)

df1.to_csv('testv2.csv')