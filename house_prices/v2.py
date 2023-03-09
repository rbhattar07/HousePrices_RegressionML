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
from sklearn.model_selection import train_test_split, cross_val_score

train_inputs = pd.read_csv('train_inputs.csv')
train_inputs.drop(columns='Unnamed: 0', inplace=True)
val_inputs = pd.read_csv('val_inputs.csv')
val_inputs.drop(columns='Unnamed: 0', inplace=True)
train_target = pd.read_csv('train_target.csv')
train_target.drop(columns='Unnamed: 0', inplace=True)
val_target = pd.read_csv('val_target.csv')
val_target.drop(columns='Unnamed: 0', inplace=True)

print(train_inputs.isna().sum())
print(val_inputs.isna().sum())

print(train_inputs.shape)
print(train_target.shape)
print(val_inputs.shape)
print(val_target.shape)


def evaluation(targets, predictions):
    mae= mean_absolute_error(targets, predictions)
    mse= mean_squared_error(targets, predictions)
    mxe= max_error(targets, predictions)
    r2s = r2_score(targets, predictions)
    mape = mean_absolute_percentage_error(targets, predictions)
    return mae, mse, mxe, r2s, mape


# Linear Regression
lr = LinearRegression().fit(train_inputs, train_target)
val_preds = lr.predict(val_inputs)
print('Linear Regression: ')
mae, mse, mxe, r2s, mape = evaluation(val_target, val_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# Ridge Regression
ridge = Ridge().fit(train_inputs, train_target)
val_preds = ridge.predict(val_inputs)
print('Ridge Regression: ')
mae, mse, mxe, r2s, mape = evaluation(val_target, val_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# Lasso Regression
lasso = Lasso().fit(train_inputs, train_target)
val_preds = lasso.predict(val_inputs)
print('Lasso Regression: ')
mae, mse, mxe, r2s, mape = evaluation(val_target, val_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# Elastic Net
en = ElasticNet().fit(train_inputs, train_target)
val_preds = en.predict(val_inputs)
print('Elastic Net: ')
mae, mse, mxe, r2s, mape = evaluation(val_target, val_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# Random Forest Regressor
rfr = RandomForestRegressor().fit(train_inputs, train_target)
val_preds = rfr.predict(val_inputs)
print('Random Forest Regressor: ')
mae, mse, mxe, r2s, mape = evaluation(val_target, val_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# Support Vector Machines
svr = SVR().fit(train_inputs, train_target)
val_preds = svr.predict(val_inputs)
print('Support Vector Machines: ')
mae, mse, mxe, r2s, mape = evaluation(val_target, val_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# XGB Regressor
xgb = XGBRegressor(min_child_weight=3).fit(train_inputs, train_target)
val_preds = xgb.predict(val_inputs)
print('XGB Regressor: ')
mae, mse, mxe, r2s, mape = evaluation(val_target, val_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

dict1 = {
    'lr':lr,
    'ridge':ridge,
    'lasso':lasso,
    'en': en,
    'rfr':rfr,
    'svm':svr,
    'xgb':xgb
}

joblib.dump(dict1, 'dict1.joblib')