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

train_inputs = pd.read_csv('trainv2.csv')
test_inputs = pd.read_csv('testv2.csv')
train_targets = pd.read_csv('train.csv')
train_targets = train_targets['SalePrice']

def evaluation(targets, predictions):
    mae= mean_absolute_error(targets, predictions)
    mse= mean_squared_error(targets, predictions)
    mxe= max_error(targets, predictions)
    r2s = r2_score(targets, predictions)
    mape = mean_absolute_percentage_error(targets, predictions)
    return mae, mse, mxe, r2s, mape

train_targets.drop(index=5, inplace=True)
train_inputs.drop(index=5, inplace=True)
