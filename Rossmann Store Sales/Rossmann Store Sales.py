__author__ = 'qiushi'

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

'''
train_set: original dataset
train_data: training dataset, X
train_label: training label set, y
cv_data: cross validation dataset, cv_x
cv_label: cross validation label set, cv_y

test_data: test dataset, test_X
result: result set with id
'''


# transfer date to year month and date in integer
# data_set: source data set
# date_index_name: the index name of date in datasets
# split_x: the character to split date
# default format of date: year/month/day
# this will delete original date in the dataset, and return a new dataset


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return rmspe


def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return "rmspe", rmspe


def trains_date(data_set, date_index_name, split_c):
    s_date = data_set[date_index_name].str.split(split_c)
    int_year = s_date.apply(lambda x: int(x[0]))
    int_month = s_date.apply(lambda x: int(x[1]))
    int_day = s_date.apply(lambda x: int(x[2]))
    int_df_date = pd.DataFrame({'year': int_year, 'Month': int_month, 'Day': int_day})
    data_set = int_df_date.join(data_set, how='outer').drop([date_index_name], axis=1)
    return data_set


# convert category to code
def encode_category(data_set, index_list):
    category_list = pd.DataFrame()
    for index in index_list:
        data_set[index] = pd.Categorical.from_array(data_set[index]).codes
    return data_set


# do this on original training set
def add_open_feature(data_set):
    df = data_set.sort_values(by=["Store", "Date"])
    count_close_days = 0
    close_day_before = []
    open_index = -1

    for index in data_set.columns:
        open_index += 1
        if index == 'Open':
            break

    if df["Open"].iloc[0] == 0:
        flag = True  # there is 0 ahead
    else:
        flag = False

    for i in df.values:
        if i[open_index] == 0:
            count_close_days += 1
            flag = True
            close_day_before.append(0)
        else:
            if flag:
                close_day_before.append(count_close_days)
                count_close_days = 0
                flag = False
            else:
                close_day_before.append(count_close_days)
    df["close_days"] = close_day_before
    return pd.DataFrame(df)


print("loading data...")
train_set = pd.read_csv("C:/Users/qiushi/OneDrive/kaggle/Rossmann Store Sales/train.csv")
test_set = pd.read_csv("C:/Users/qiushi/OneDrive/kaggle/Rossmann Store Sales/test.csv")
store_info = pd.read_csv("C:/Users/qiushi/OneDrive/kaggle/Rossmann Store Sales/store.csv")

print("processing data...")
train_set = add_open_feature(train_set)
test_set = add_open_feature(test_set)

train_set = pd.DataFrame(train_set).merge(store_info)
test_set = pd.DataFrame(test_set).merge(store_info)
train_set = train_set[train_set['Open'] == 1]
# category index
index_list = ['StoreType', 'Promo', 'Assortment', 'PromoInterval']

# select features
train_data = pd.concat([train_set['DayOfWeek'], train_set['Promo'], train_set['Date'],
                        train_set['StoreType'], train_set['Assortment'], train_set['CompetitionDistance'],
                        train_set['CompetitionOpenSinceYear'], train_set['CompetitionOpenSinceMonth'],
                        train_set['SchoolHoliday'], train_set['Promo2'], train_set['Promo2SinceWeek'],
                        train_set['Promo2SinceYear'], train_set['PromoInterval'], train_set['Store'],
                        train_set['close_days']
                        ], axis=1,
                       keys=['DayOfWeek', 'Promo', 'Date',
                             'StoreType', 'Assortment', 'CompetitionDistance',
                             'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth',
                             'SchoolHoliday',
                             'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'Store', 'close_days'])

# train_data = train_set
# train_data = train_data.drop(['Sales', 'Customers', 'Open'], axis=1)
train_label = train_set['Sales']

# fill nan with 0
train_data = train_data.fillna(0)
# split and transfer date to integer
train_data = trains_date(train_data, 'Date', '-')
# transfer category
train_data = encode_category(train_data, index_list)


# separate data to train and CV
X, cv_x, Y, cv_y = train_test_split(train_data, train_label, test_size=0.1, random_state=85)
# X = train_data
# Y = train_label

# build model
print('building model...')
print('random forest..')
rfr = RandomForestRegressor(n_estimators=60, random_state=30)
rfr.fit(X, np.log(Y + 1))
# rfr_score = rfr.score(cv_x, cv_y)
# print('rfr score:', rfr_score)

importance = rfr.feature_importances_

print('finished')
print('importance', importance)

# predicting
print('precessing test data...')

test_data = pd.concat([test_set['DayOfWeek'], test_set['Promo'], test_set['Date'], test_set['Id'], test_set['Open'],
                       test_set['StoreType'], test_set['Assortment'], test_set['CompetitionDistance'],
                       test_set['CompetitionOpenSinceYear'], test_set['CompetitionOpenSinceMonth'],
                       test_set['SchoolHoliday'], test_set['Promo2'], test_set['Promo2SinceWeek'],
                       test_set['Promo2SinceYear'], test_set['PromoInterval'], test_set['Store'], test_set['close_days']
                       ], axis=1,
                      keys=['DayOfWeek', 'Promo', 'Date', 'Id', 'Open',
                            'StoreType', 'Assortment', 'CompetitionDistance',
                            'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth',
                            'SchoolHoliday',
                            'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'Store', 'close_days'])

# test_data = test_data.drop(['Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'], asix=1)
# fill na with 0
test_data = test_data.fillna(0)
# transfer date to year month and date
test_data = trains_date(test_data, 'Date', '-')
# convert category to code
test_data = encode_category(test_data, index_list)

# cv
cv_sales = rfr.predict(cv_x.values)
error = rmspe(np.exp(cv_sales) - 1, cv_y)
print('error', error)

print('predicting...')
result = pd.DataFrame({"Id": test_data.Id, "Sales": 0, "Open": test_data.Open})

result.Sales = np.exp(rfr.predict(test_data.drop(['Id', 'Open'], axis=1).values)) + 1
result_y = result.apply(lambda x: 0 if x.Open == 0 else x.Sales, axis=1)
# write into CSV
print('writing to csv...')
pd.DataFrame({"Id": result.Id, "Sales": result_y}).to_csv(
    'C:\\Users\\qiushi\\OneDrive\\kaggle\\Rossmann Store Sales\\submit.csv', index=False, header=True)

print("all done.")
