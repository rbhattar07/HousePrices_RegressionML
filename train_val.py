import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

train_df = pd.read_csv('train_df.csv')
val_df = pd.read_csv('val_df.csv')

# Defining input and target columns
input_cols = list(train_df.columns)[2:-1]
target_cols = 'SalePrice'

# Creating train & Val inputs and targets
train_inputs = train_df[input_cols].copy()
train_target = train_df[target_cols].copy()
val_inputs = val_df[input_cols].copy()
val_target = val_df[target_cols].copy()

# Defining numerical & Categorical columns
numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_df.select_dtypes('object').columns.tolist()
