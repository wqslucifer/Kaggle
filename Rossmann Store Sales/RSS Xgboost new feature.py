__author__ = 'qiushi'

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import cross_validation
import xgboost as xgb


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


# add new feature
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
print("train set")
train_set = pd.DataFrame(train_set).merge(store_info)
test_set = pd.DataFrame(test_set).merge(store_info)

# add new feature: if the store closed yesterday
train_set = add_open_feature(train_set)
# train_set.to_csv("C:/Users/qiushi/OneDrive/kaggle/Rossmann Store Sales/train_with_new_feature.csv")
train_set = train_set[train_set['Open'] == 1]
# category index
index_list = ['StoreType', 'Promo', 'Assortment', 'PromoInterval']

print("select features")
# select features
train_data = pd.concat([train_set['DayOfWeek'], train_set['Promo'], train_set['Date'],
                        train_set['StoreType'], train_set['Assortment'], train_set['CompetitionDistance'],
                        train_set['CompetitionOpenSinceYear'], train_set['CompetitionOpenSinceMonth'],
                        train_set['SchoolHoliday'], train_set['Promo2'], train_set['Promo2SinceWeek'],
                        train_set['Promo2SinceYear'], train_set['PromoInterval'], train_set['close_days'],
                        train_set['Store']
                        ], axis=1,
                       keys=['DayOfWeek', 'Promo', 'Date',
                             'StoreType', 'Assortment', 'CompetitionDistance',
                             'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth',
                             'SchoolHoliday',
                             'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'close_days', 'Store'])
train_label = train_set['Sales']

# fill nan with 0
train_data = train_data.fillna(0)
# split and transfer date to integer
train_data = trains_date(train_data, 'Date', '-')
# transfer category
train_data = encode_category(train_data, index_list)

print("test set")
test_set = add_open_feature(test_set)
test_data = pd.concat([test_set['DayOfWeek'], test_set['Promo'], test_set['Date'], test_set['Id'], test_set['Open'],
                       test_set['StoreType'], test_set['Assortment'], test_set['CompetitionDistance'],
                       test_set['CompetitionOpenSinceYear'], test_set['CompetitionOpenSinceMonth'],
                       test_set['SchoolHoliday'], test_set['Promo2'], test_set['Promo2SinceWeek'],
                       test_set['Promo2SinceYear'], test_set['PromoInterval'], test_set['close_days'], test_set['Store']
                       ], axis=1,
                      keys=['DayOfWeek', 'Promo', 'Date', 'Id', 'Open',
                            'StoreType', 'Assortment', 'CompetitionDistance',
                            'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth',
                            'SchoolHoliday',
                            'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'close_days', 'Store'])
# fill na with 0
test_data = test_data.fillna(0)
# transfer date to year month and date
test_data = trains_date(test_data, 'Date', '-')
# convert category to code
test_data = encode_category(test_data, index_list)

X, cv_x, Y, cv_y = cross_validation.train_test_split(train_data, train_label, test_size=0.01, random_state=85)

# build model
print('building model...')

print('xgboost..')
params = {"objective": "reg:linear",
          "eta": 0.25,
          "max_depth": 9,
          "subsample": 0.9,
          "colsample_bytree": 0.7,

          "silent": 1
          }
num_trees = 600

d_train = xgb.DMatrix(X.values, label=np.log(Y + 1))
d_cv = xgb.DMatrix(cv_x.values, label=np.log(cv_y + 1))

watchlist = [(d_cv, 'cv'), (d_train, 'train')]
gbm = xgb.train(params, d_train, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg,
                verbose_eval=True)

print("xgboost validating")
train_prob = gbm.predict(xgb.DMatrix(cv_x.values))
indices = train_prob < 0
train_prob[indices] = 0
error = rmspe(np.exp(train_prob) - 1, cv_y)
print('error', np.float(error))

print('predicting...')
print('xgboost...')

result = pd.DataFrame({"Id": test_data.Id, "Sales": 0, "Open": test_data.Open})
result.loc[test_data['Open'] == 1, 'Sales'] = [np.exp(x) - 1 for x in gbm.predict(xgb.DMatrix(test_data.values))]

print('writing to csv...')
result.to_csv("C:\\Users\\qiushi\\OneDrive\\kaggle\\Rossmann Store Sales\\xgboost_submit.csv", index=False)
