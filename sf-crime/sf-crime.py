__author__ = 'qiushi'

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt


# load data
train_data = pd.read_csv("C:\\Users\\qiushi\OneDrive\kaggle\sf-crime\\train.csv")
test_data = pd.read_csv("C:\\Users\\qiushi\OneDrive\kaggle\sf-crime\\test.csv")


def transform_date(train_date):
    date_int = pd.DataFrame(columns=['year', 'month', 'day', 'time'])
    row, col = train_data.shape
    j = 0
    for i in train_date:
        print("job=", round(j / row * 100, 2), "%")
        j += 1
        date, time = i.split(' ')
        y, m, d = date.split('-')
        time = time.split(':')[:2]
        time = int(time[0]) * 60 + int(time[1])
        date_int = date_int.append(pd.Series([y, m, d, time], index=['year', 'month', 'day', 'time']),
                                   ignore_index=True)
    return date_int


# Recode categories to numerical
X = pd.DataFrame()
y = pd.Categorical.from_array(train_data['Category']).codes
X.DayOfWeek = pd.Categorical.from_array(train_data['DayOfWeek']).codes
X.PdDistrict = pd.Categorical.from_array(train_data['PdDistrict']).codes
print("transfor date...")
date = transform_date(train_data.Dates)
print(date[:10])
'''
vectorizer = DictVectorizer(sparse=False)
vec_train = vectorizer.fit_transform(category_dict)
print(vec_train)
test = vectorizer.transform("OTHER OFFENSES")
'''
