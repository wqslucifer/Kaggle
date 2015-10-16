__author__ = 'qiushi'

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize
import numpy as np

print("loading data...")
train_set = pd.read_csv("C:/Users/qiushi/OneDrive/kaggle/Rossmann Store Sales/train.csv")
test_set = pd.read_csv("C:/Users/qiushi/OneDrive/kaggle/Rossmann Store Sales/test.csv")
store_info = pd.read_csv("C:/Users/qiushi/OneDrive/kaggle/Rossmann Store Sales/store.csv")

print("processing data...")
train_set = pd.DataFrame(train_set).merge(store_info)
test_set = pd.DataFrame(test_set).merge(store_info)
train_set = train_set[train_set['Open'] == 1]

# select features
train_data = pd.concat([train_set['DayOfWeek'], train_set['Promo'],
                        train_set['StoreType'], train_set['Assortment']], axis=1,
                       keys=['DayOfWeek', 'Promo', 'StoreType', 'Assortment'])
# transfer date to year month and date
date = train_set.Date.str.split('-')
year = date.apply(lambda x: int(x[0]))
month = date.apply(lambda x: int(x[1]))
day = date.apply(lambda x: int(x[2]))
df_date = pd.DataFrame({'year': year, 'Month': month, 'Date': day})
train_data = df_date.join(train_data, how='outer')

train_label = train_set['Sales']
train_customers = train_set['Customers']
# convert category to code
train_data.Promo = pd.Categorical.from_array(train_data['Promo']).codes
train_data.StoreType = pd.Categorical.from_array(train_data['StoreType']).codes
train_data.Assortment = pd.Categorical.from_array(train_data['Assortment']).codes
# train_data.CompetitionOpenSinceYear.fillna(0)
# train_data.CompetitionOpenSinceYear = pd.Categorical.from_array(train_data['CompetitionOpenSinceYear']).codes

# separate data to train and CV
# X, cv_x, Y, cv_y = train_test_split(train_data, train_label, test_size=0.30, random_state=85)
X = train_data
Y = train_label

# build model
print('building model...')
print('random forest..')
rfr = RandomForestRegressor(n_estimators=70)
rfr.fit(X, Y)
# rfr_score = rfr.score(cv_x, cv_y)
# print('rfr score:', rfr_score)

print('finished')

# predicting
print('precessing test data...')
test_data = pd.concat([test_set['DayOfWeek'], test_set['Promo'], test_set['Id'], test_set['Open'],
                       test_set['StoreType'], test_set['Assortment'],
                       test_set['CompetitionOpenSinceYear']], axis=1,
                      keys=['DayOfWeek', 'Promo', 'Id', 'Open',
                            'StoreType', 'Assortment'])
# transfer date to year month and date


# convert category to code
test_data.Promo = pd.Categorical.from_array(test_set['Promo']).codes
test_data.StoreType = pd.Categorical.from_array(test_set['StoreType']).codes
test_data.Assortment = pd.Categorical.from_array(test_set['Assortment']).codes
# test_data.CompetitionOpenSinceYear = test_data.CompetitionOpenSinceYear.fillna(0)
# test_data.CompetitionOpenSinceYear = pd.Categorical.from_array(test_set['CompetitionOpenSinceYear']).codes
# transfer date to year month and date
date = test_set.Date.str.split('-')
year = date.apply(lambda x: int(x[0]))
month = date.apply(lambda x: int(x[1]))
day = date.apply(lambda x: int(x[2]))
df_date = pd.DataFrame({'year': year, 'Month': month, 'Date': day})
test_data = df_date.join(test_data, how='outer')

print('predicting...')
result = pd.DataFrame({"Id": test_data.Id, "Sales": 0, "Open": test_data.Open})

result.Sales = rfr.predict(test_data.drop(['Id', 'Open'], axis=1).values)
result.set_index("Id")
result = result.sort_index(axis=1)
result_y = result.apply(lambda x: 0 if x.Open == 0 else x.Sales, axis=1)
# write into CSV
print('writing to csv...')
pd.DataFrame({"Id": result.Id, "Sales": result_y}).to_csv(
    'C:\\Users\\qiushi\\OneDrive\\kaggle\\Rossmann Store Sales\\submit.csv', index=False, header=True)

print("all done.")
