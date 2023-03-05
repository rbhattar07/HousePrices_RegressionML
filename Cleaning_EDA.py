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

df = raw_df.copy()
df.drop_duplicates(inplace=True)
df.drop(columns=['MiscFeature', 'Fence', 'PoolQC', 'Alley'], inplace=True)

# Setting numeric and categorical columns
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
categorical_columns = df.select_dtypes('object').columns.tolist()


# EDA- Exploratory Data Analysis

print('Numeric Columns:')
print(df[numeric_columns].columns)
print('Categorical Columns:')
print(df[categorical_columns].columns)

fig = px.scatter(df, x='YrSold',
                 y='MoSold',
                 color='SaleCondition',
                 opacity=0.5,
                 hover_data=['SalePrice'],
                 title='Yrsold vs Mosold').update_traces(marker_size=6)
fig.show()

fig = px.scatter(df, x='LotArea',
                 y='SalePrice',
                 color='SaleCondition',
                 opacity=0.5,
                 hover_data=['YrSold'],
                 title='Lot area vs Sale Price').update_traces(marker_size=6)
fig.show()

fig = px.scatter(df, x='LotArea',
                 y='LotFrontage',
                 color='GarageType',
                 opacity=0.5,
                 hover_data=['SalePrice']).update_traces(marker_size=6)
fig.show()

fig = px.scatter(df, x='GarageYrBlt',
                 y='GarageArea',
                 color='GarageCond',
                 opacity=0.5,
                 hover_data=['SalePrice'],
                 title='Lot are vs lot frontage').update_traces(marker_size=6)
fig.show()

fig = px.scatter(df, x='GarageYrBlt',
                 y='GarageArea',
                 color='GarageCond',
                 opacity=0.5,
                 hover_data=['SalePrice'],
                 title='Lot are vs lot frontage').update_traces(marker_size=6)
fig.show()

fig = px.histogram(df,
                   x='SalePrice',
                   marginal='box',
                   nbins = 25,
                   title='Distribution of SalePrice').update_layout(bargap=0.1)
fig.show()

fig = px.scatter(df, x='FullBath',
                 y='HalfBath',
                 color='Neighborhood',
                 opacity=0.5,
                 hover_data=['SalePrice'],
                 title='Full Bath vs Half Bath').update_traces(marker_size=6)
fig.show()

fig = px.scatter(df, x='BsmtFullBath',
                 y='BsmtHalfBath',
                 color='Neighborhood',
                 opacity=0.5,
                 hover_data=['SalePrice'],
                 title='Basement full vs half bath').update_traces(marker_size=6)
fig.show()

fig = px.scatter(df, x='YearBuilt',
                 y='YearRemodAdd',
                 color='Neighborhood',
                 opacity=0.5,
                 hover_data=['SalePrice'],
                 title='Yearbuilt vs year remodelling in NEIGHBORHOOD').update_traces(marker_size=6)
fig.show()

fig = px.histogram(df,
                   x='Neighborhood',
                   y='SalePrice',
                   marginal='box',
                   nbins = 25,
                   title='Distribution of Neighborhood & SalePrice').update_layout(bargap=0.1)
fig.show()

fig = px.scatter(df, x='OverallQual',
                 y='OverallCond',
                 color='SalePrice',
                 opacity=0.5,
                 hover_data=['Neighborhood'],
                 title='Neighborhood wise condition & Quality').update_traces(marker_size=6)
fig.show()

fig = px.scatter(df, x='OverallQual',
                 y='SalePrice',
                 color='Neighborhood',
                 opacity=0.5,
                 title='Overall Quality vs Saleprice').update_traces(marker_size=6)
fig.show()

fig = px.scatter(df, x='OverallCond',
                 y='SalePrice',
                 color='Neighborhood',
                 opacity=0.5,
                 title='Overall Condition vs Sale pRICE').update_traces(marker_size=6)
fig.show()

for i in df[categorical_columns]:
    print(df[i].value_counts())

# Creating training and validation sets
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.60, random_state=42)

train_df.to_csv('train_df.csv')
val_df.to_csv('val_df.csv')
