__author__ = 'qiushi'

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# load data
print("loading data...")
train_data = pd.read_csv("C:\\Users\\qiushi\OneDrive\kaggle\sf-crime\\train.csv")
test_data = pd.read_csv("C:\\Users\\qiushi\OneDrive\kaggle\sf-crime\\test.csv")

'''
train_data = train_data[:20]
test_data = test_data[:20]
'''


def transform_date(train):
    # separate the date to year, month, date and time in int
    train.DayOfWeek = pd.Categorical.from_array(train['DayOfWeek']).codes
    train.PdDistrict = pd.Categorical.from_array(train['PdDistrict']).codes
    df_date_time = train.Dates.str.split(' ')
    df_date = df_date_time.apply(lambda x: x[0])
    df_time = df_date_time.apply(lambda x: x[1])

    df_sp_date = df_date.str.split('-')
    df_year_int = df_sp_date.apply(lambda x: int(x[0]))
    df_month_int = df_sp_date.apply(lambda x: int(x[1]))
    df_date_int = df_sp_date.apply(lambda x: int(x[2]))

    df_sp_time = df_time.str.split(':')
    df_hours_int = df_sp_time.apply(lambda x: int(x[0]))
    df_minutes_int = df_sp_time.apply(lambda x: int(x[1]))
    df_seconds_int = df_sp_time.apply(lambda x: int(x[2]))
    time_int = df_sp_time.apply(lambda x: int(x[0]) * 60 + int(x[1]))
    # time is in minutes without seconds
    date_int = pd.DataFrame({'Year': df_year_int, 'Month': df_month_int,
                             'Date': df_date_int, 'Time': time_int})
    train = date_int.join(train.drop('Dates', axis=1), how='outer')
    # separate address into category
    # train.Address.apply(lambda x: x.split('/') if '/' in x else x.split('of'))

    return train


# main code
# Recode categories to numerical
print("transform data into proper type...")
category_train = pd.Categorical.from_array(train_data['Category']).categories
train_data.Category = pd.Categorical.from_array(train_data['Category']).codes

X = transform_date(train_data)
y = X.Category
X = X.drop(['Descript', 'Resolution', 'Address', 'Category'], axis=1)
print("data transformation completed")

# fit the model
print("fitting...")
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X, y)

# make prediction
print("transform test_data")
test_data = transform_date(test_data)
test_data = test_data.drop(['Address', 'Id'], axis=1)
print("predicting...")
predict = rf.predict(test_data)
predict_str = pd.Categorical.from_codes(predict, category_train)
print("finished")

# write into CSV
predict_str = pd.Series(predict_str[:20])
result = pd.DataFrame(predict_str.apply(lambda x: (x == category_train).astype(int)))
print(category_train);
result[:20].to_csv("C:\\Users\qiushi\OneDrive\kaggle\\sf-crime\\submit.csv", index=False, header=category_train)
