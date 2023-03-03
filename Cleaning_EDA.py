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

df = pd.read_csv(train_csv_path)
print(df.info())
print(' ')
print('Statistical Description: ')
print(df.describe())

print(df.columns)

numeric_data = df.select_dtypes(include=np.number)
categorical_data = df.select_dtypes('object')

print(numeric_data)
print(categorical_data)

print(numeric_data.columns)
print(categorical_data.columns)

# Year Built
fig = px.histogram(df, x='YearBuilt', marginal='box', nbins= 20, title='Distribution of Year Built')
fig.update_layout(bargap=0.1)
fig.show()

# Year Built & Year Sold
fig=px.scatter(df, marginal_y='box', x='YearBuilt', y='YrSold',color='SalePrice',opacity=0.5, title='Built year vs Sale Year')
fig.update_traces(marker_size = 8)
fig.show()

# Year sold vs month sold with Overall Quality
fig=px.scatter(df, x='YrSold', y='MoSold', color='OverallQual',
           opacity=0.5, title='Year vs Month of Sales').update_traces(marker_size=5)
fig.show()