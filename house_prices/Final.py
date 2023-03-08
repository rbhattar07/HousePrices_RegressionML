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

def evaluation(targets, predictions):
    mae= mean_absolute_error(targets, predictions)
    mse= mean_squared_error(targets, predictions)
    mxe= max_error(targets, predictions)
    r2s = r2_score(targets, predictions)
    mape = mean_absolute_percentage_error(targets, predictions)
    return mae, mse, mxe, r2s, mape

df = pd.read_csv('testv2.csv')
target = pd.read_csv('train.csv')
target = target['SalePrice']
target.drop(index=5, inplace=True)

# Linear Regression
lr = LinearRegression().fit(df, target)
lr_preds= lr.predict(df)

print('Linear Regression:')
mae, mse, mxe, r2s, mape = evaluation(target, lr_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)

# Ridge Regression
ridge = Ridge(solver='lbfgs', positive=True).fit(train_inputs, train_target)
train_preds= ridge.predict(train_inputs)
val_preds= ridge.predict(val_inputs)

print('Ridge:')
print('Val evaluation:')
mae, mse, mxe, r2s, mape = evaluation(val_target, val_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# Lasso Regression
lasso = Lasso(max_iter=1000, precompute=False).fit(train_inputs, train_target)
train_preds= lasso.predict(train_inputs)
val_preds= lasso.predict(val_inputs)

print('lasso:')
print('Val evaluation:')
mae, mse, mxe, r2s, mape = evaluation(val_target, val_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# Elastic Net
en = ElasticNet(l1_ratio=1, max_iter=2500).fit(train_inputs, train_target)
train_preds= en.predict(train_inputs)
val_preds= en.predict(val_inputs)

print('Elastic Net:')
print('Val evaluation:')
mae, mse, mxe, r2s, mape = evaluation(val_target, val_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# Random Forest Regressor
rfr = RandomForestRegressor(n_estimators=110, min_samples_leaf=4, min_impurity_decrease=250, verbose=1).fit(train_inputs, train_target)
train_preds= rfr.predict(train_inputs)
val_preds= rfr.predict(val_inputs)

print('Random Forest Regressor')
print('Val evaluation:')
mae, mse, mxe, r2s, mape = evaluation(val_target, val_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# Support Vector Machines
svr = SVR(kernel='linear').fit(train_inputs, train_target)
train_preds= svr.predict(train_inputs)
val_preds= svr.predict(val_inputs)

print('SVR')
print('Val evaluation:')
mae, mse, mxe, r2s, mape = evaluation(val_target, val_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)

# XGboost Regressor
xgb = XGBRegressor(learning_rate=0.2, max_depth=4).fit(train_inputs, train_target)
train_preds= xgb.predict(train_inputs)
val_preds= xgb.predict(val_inputs)

print('XGBRegressor')
print('Val evaluation:')
mae, mse, mxe, r2s, mape = evaluation(val_target, val_preds)
print('MAE',mae)
print('MSE',mse)
print('MXE',mxe)
print('r2s',r2s)
print('MAPE',mape)
print('-'*30)
