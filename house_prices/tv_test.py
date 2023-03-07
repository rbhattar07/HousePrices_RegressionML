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
from sklearn.preprocessing import PolynomialFeatures

tv = joblib.load('TV_models.joblib')
df = joblib.load('train_val.joblib')
test_df = pd.read_csv('test.csv')
# loading elements
train_inputs = df['train_inputs']
val_inputs = df['val_inputs']
train_target = df['train_target']
val_target = df['val_target']
scaler = df['scaler']
imputer = df['imputer']
encoder = df['encoder']
input_cols = df['input_columns']
target_cols = df['target_columns']
numeric_cols= df['numeric_columns']
categorical_cols= df['categorical_columns']
encoded_cols = df['encoded_columns']
# Loading Models
lr = tv['Linear_Regression']
ridge = tv['ridge']
lasso = tv['lasso']
en= tv['elastic_net']
rfr = tv['randomforestregressor']
svr = tv['svr']
xgbregressor = tv['xgbregressor']
train_target = tv['train_target']
val_target = tv['val_target']

# imputing missing data
test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])
# Scaling numeric features
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
# Encoding Categorical Columns
test_df[encoded_cols] = encoder.transform(test_df[categorical_cols])

test_df.to_csv('updated_test_df.csv')