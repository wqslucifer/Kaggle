__author__ = 'qiushi'

import numpy as np;
import pandas as pd;
from sklearn.cross_validation import train_test_split;

print("loading data...")
train_set = pd.read_csv("C:/Users/qiushi/OneDrive/kaggle/Rossmann Store Sales/train.csv")
test_set = pd.read_csv("C:/Users/qiushi/OneDrive/kaggle/Rossmann Store Sales/test.csv")
store_info = pd.read_csv("C:/Users/qiushi/OneDrive/kaggle/Rossmann Store Sales/store.csv")

m = train_set.shape[0]

print("processing data...")
train_set = pd.DataFrame(train_set).merge(store_info)
test_set = pd.DataFrame(test_set).merge(store_info)
train_set = train_set[train_set['Open'] == 1]

train_data = pd.concat([train_set['DayOfWeek'], train_set['Customers'],
                        train_set['Promo'], train_set['StoreType'], train_set['Assortment'],
                        train_set['CompetitionOpenSinceYear']], axis=1,
                       keys=['DayOfWeek', 'Customers', 'Promo', 'StoreType', 'Assortment',
                             'CompetitionOpenSinceYear'])
train_label = train_set['Sales']

X, cv_x, Y, cv_y = train_test_split(train_set, train_label, test_size=0.30, random_state=25)

print('X', X.shape)
print('Y', Y.shape)
print('cv_x', cv_x.shape)
print('cv_y', cv_y.shape)
