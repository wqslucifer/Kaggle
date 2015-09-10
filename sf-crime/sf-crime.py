__author__ = 'qiushi'

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt


# load data
train_data = pd.read_csv("C:\\Users\\qiushi\OneDrive\kaggle\sf-crime\\train.csv")
test_data = pd.read_csv("C:\\Users\\qiushi\OneDrive\kaggle\sf-crime\\test.csv")


def transform_date(train):
    # separate the date to year, month, date and time in int
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
    train.Address.apply(lambda x: x.split('/') if '/' in x else x.split('of'))

    return train


# Recode categories to numerical
train_data.Category = pd.Categorical.from_array(train_data['Category']).codes
train_data.DayOfWeek = pd.Categorical.from_array(train_data['DayOfWeek']).codes
train_data.PdDistrict = pd.Categorical.from_array(train_data['PdDistrict']).codes
X = transform_date(train_data)
print(X[:10])
'''
vectorizer = DictVectorizer(sparse=False)
vec_train = vectorizer.fit_transform(category_dict)
print(vec_train)
test = vectorizer.transform("OTHER OFFENSES")
'''
