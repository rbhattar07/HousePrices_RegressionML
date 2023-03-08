import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures

test_df = pd.read_csv('test.csv')
models = joblib.load('TV_models.joblib')
elements = joblib.load('train_val.joblib')

imputer = elements['imputer']
scaler = elements['scaler']
encoder = elements['encoder']

print(test_df.info())
test_df.drop(columns=['MiscFeature', 'Fence', 'PoolQC', 'Alley', 'FireplaceQu', 'Id'], inplace=True)

# Defining Numeric, Categorical & Encoded Columns
numerical_cols = test_df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = test_df.select_dtypes('object').columns.tolist()

# Imputing Missing Numerical data
test_df[numerical_cols] = imputer.transform(test_df[numerical_cols])
# Scaling Numeric Features
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])
# Encoding Categorical Features
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
test_df[encoded_cols]=encoder.transform(test_df[categorical_cols])

test_df.drop(columns=categorical_cols, inplace=True)

test_df.to_csv('utestdf.csv')