import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

pd.options.display.max_columns = 200
pd.options.display.max_rows = 200
sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=14
matplotlib.rcParams['figure.figsize']=(10,8)
matplotlib.rcParams['figure.facecolor']='#00000000'

data_dir = 'house_prices'
print(os.listdir(data_dir))
train_csv_path = data_dir + '/train.csv'

raw_df = pd.read_csv(train_csv_path)
print(raw_df)
print(raw_df.info())
print(raw_df.describe())

# Setting numeric and categorical columns
numeric_columns = raw_df.select_dtypes(include=np.number).columns.tolist()
categorical_columns = raw_df.select_dtypes('object').columns.tolist()

print('Missing numeric columns')
print(raw_df[numeric_columns].isna().sum())
print('Missing Categorical Columns')
print(raw_df[categorical_columns].isna().sum())

df = raw_df.copy()
df.drop_duplicates(inplace=True)