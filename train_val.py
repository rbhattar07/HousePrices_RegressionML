import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

train_df = pd.read_csv('train_df.csv')
val_df = pd.read_csv('val_df.csv')

# Defining input and target columns
input_cols = list(train_df.columns)[1:-1]
target_cols = 'SalePrice'

# Creating train & Val inputs and targets
train_inputs = train_df[input_cols].copy()
train_target = train_df[target_cols].copy()
val_inputs = val_df[input_cols].copy()
val_target = val_df[target_cols].copy()

# Defining numerical & Categorical columns
numerical_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

# Imputing missing numerical data
print(train_inputs[numerical_cols].isna().sum())
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
imputer.fit(train_inputs[numerical_cols])
train_inputs[numerical_cols] = imputer.transform(train_inputs[numerical_cols])
val_inputs[numerical_cols] = imputer.transform(val_inputs[numerical_cols])

# Scaling Numeric Features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_inputs[numerical_cols])
train_inputs[numerical_cols]= scaler.transform(train_inputs[numerical_cols])
val_inputs[numerical_cols]=scaler.transform(val_inputs[numerical_cols])

# Encoding Categorical data
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(train_inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])

# Model Building
#----- LR model
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(train_inputs[numerical_cols+encoded_cols], train_target)
