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

lr = tv['Linear_Regression']
ridge = tv['ridge']
lasso = tv['lasso']
en = tv['elastic_net']
rfr = tv['randomforestregressor']
svr = tv['svr']
xgb = tv['xgbregressor']

# test_df
test_df = pd.read_csv('utestdf.csv')
test_df.drop(columns='Unnamed: 0', inplace=True)
train_df = pd.read_csv('train.csv')

target = train_df['SalePrice']
target = target.drop(index=1)

# Linear Regression
lr_preds = lr.predict(test_df)
# Ridge
ridge_preds = ridge.predict(test_df)
#lasso
lasso_preds = lasso.predict(test_df)
# Elastic Net
en_preds= en.predict(test_df)
# Random Forest Regressor
rfr_preds = rfr.predict(test_df)
# Support Vector Machines
svr_preds = svr.predict(test_df)
# XGBoost Regressor
xgb_preds = xgb.predict(test_df)

# Testing the efficiency of our model on test set
# Our model was made just on the basis of 584 observations and 262 columns. SO we can expect a lot of accuracy problems.
def evaluation(targets, predictions):
    mae= mean_absolute_error(targets, predictions)
    mse= mean_squared_error(targets, predictions)
    mxe= max_error(targets, predictions)
    r2s = r2_score(targets, predictions)
    mape = mean_absolute_percentage_error(targets, predictions)
    return mae, mse, mxe, r2s, mape

# Linear Regression
print('Linear Regression: ')
print('Test evaluation:')
mae, mse, mxe, r2s, mape = evaluation(target, lr_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)

# Ridge
print('Ridge:')
print('Test evaluation:')
mae, mse, mxe, r2s, mape = evaluation(target, ridge_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# Lasso
print('lasso:')
print('Val evaluation:')
mae, mse, mxe, r2s, mape = evaluation(target, lasso_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# Elastic Net
print('Elastic Net:')
print('Test evaluation:')
mae, mse, mxe, r2s, mape = evaluation(target, en_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# Random Forest Regressor
print('Random Forest:')
print('Test evaluation:')
mae, mse, mxe, r2s, mape = evaluation(target, rfr_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# Support Vector Machines
print('Support Vector Machines:')
print('Test evaluation:')
mae, mse, mxe, r2s, mape = evaluation(target, svr_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# XGboost Regressor
print('XGboost Regressor:')
print('Test evaluation:')
mae, mse, mxe, r2s, mape = evaluation(target, xgb_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)