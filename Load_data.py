from urllib.request import urlretrieve
import os
import zipfile
import kaggle
import opendatasets as od
from kaggle.api.kaggle_api_extended import KaggleApi

dataset_url = 'https://github.com/JovianML/opendatasets/raw/master/data/house-prices-advanced-regression-techniques.zip'

urlretrieve(dataset_url, 'house_prices.zip')

with zipfile.ZipFile('house_prices.zip') as f:
    f.extractall(path='house_prices')