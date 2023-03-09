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

df = pd.read_csv('train.csv')

# Creating training and validation sets
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.60, random_state=42)
train_target, val_target = train_test_split(df, test_size=0.60, random_state=42)

train_target = train_target['SalePrice']
val_target = val_target['SalePrice']

numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove('SalePrice')
numerical_cols.remove('Id')

categorical_cols = df.select_dtypes('object').columns.tolist()
categorical_cols.remove('MiscFeature')
categorical_cols.remove('Fence')
categorical_cols.remove('PoolQC')
categorical_cols.remove('FireplaceQu')
categorical_cols.remove('Alley')

selected_num_cols = list(df.corr()["SalePrice"][(df.corr()["SalePrice"]>0.50) | (df.corr()["SalePrice"]<-0.50)].index)
selected_cat_cols = [
    'MSZoning', 'Utilities', 'BldgType', 'Heating', 'SaleType', 'SaleCondition'
]
r = ['MSZoning', 'Utilities', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
    'RoofMatl', 'Exterior1st', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'Heating', 'HeatingQC',
    'CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 'SaleCondition']
imp_cols = selected_num_cols+selected_cat_cols

# Creating an updated test set, numerical & Categorical Columns
train_inputs = train_df[imp_cols]
numerical_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

val_inputs = val_df[imp_cols]

# Scaling Numeric Features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train_inputs[numerical_cols])
train_inputs[numerical_cols] = scaler.transform(train_inputs[numerical_cols])
val_inputs[numerical_cols] = scaler.transform(val_inputs[numerical_cols])


# Encoding Categorical Columns
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])

train_inputs.drop(columns=categorical_cols, inplace=True)
val_inputs.drop(columns=categorical_cols, inplace=True)

train_inputs.to_csv('train_inputs.csv')
train_target.to_csv('train_target.csv')
val_inputs.to_csv('val_inputs.csv')
val_target.to_csv('val_target.csv')

