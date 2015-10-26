__author__ = 'qiushi'

import pandas as pd;
import pickle

from sklearn.cross_validation import train_test_split
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale


# transfer date to year month and date in integer
# data_set: source data set
# date_index_name: the index name of date in datasets
# split_x: the character to split date
# default format of date: year/month/day
# this will delete original date in the dataset, and return a new dataset
def trains_date(data_set, date_index_name, split_c):
    s_date = data_set[date_index_name].str.split(split_c)
    int_year = s_date.apply(lambda x: int(x[0]))
    int_month = s_date.apply(lambda x: int(x[1]))
    int_day = s_date.apply(lambda x: int(x[2]))
    int_df_date = pd.DataFrame({'year': int_year, 'Month': int_month, 'Day': int_day})
    data_set = int_df_date.join(data_set, how='outer').drop([date_index_name], axis=1)
    return data_set


# convert category to code, using pd.Categorical.codes
def encode_category(data_set, index_list):
    category_list = pd.DataFrame()
    for index in index_list:
        data_set[index] = pd.Categorical.from_array(data_set[index]).codes
    return data_set


# OneHotEncode category
def OneHotEncode_category(data_set, index_list):
    enc = OneHotEncoder()
    for i in index_list:
        enc.fit(train_data[i])
        train_data[i] = enc.transform(train_data[i])
    return train_data


print("loading data...")
train_set = pd.read_csv("C:/Users/qiushi/OneDrive/kaggle/Rossmann Store Sales/train.csv")
test_set = pd.read_csv("C:/Users/qiushi/OneDrive/kaggle/Rossmann Store Sales/test.csv")
store_info = pd.read_csv("C:/Users/qiushi/OneDrive/kaggle/Rossmann Store Sales/store.csv")

print("processing data...")
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
                        train_set['Promo2SinceYear'], train_set['PromoInterval']
                        ], axis=1,
                       keys=['DayOfWeek', 'Promo', 'Date',
                             'StoreType', 'Assortment', 'CompetitionDistance',
                             'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth',
                             'SchoolHoliday',
                             'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'])
train_label = train_set['Sales']
# fill nan with 0
train_data = train_data.fillna(0)
# split and transfer date to integer
train_data = trains_date(train_data, 'Date', '-')
# encode
train_data = encode_category(train_data, index_list)


# X, cv_x, Y, cv_y = train_test_split(train_data, train_label, test_size=0.30, random_state=85)
X = train_data
Y = train_label

print("building dataset")
# build model
input_size = X.shape[1]
target_size = 1

# supervised dataset
ds = SupervisedDataSet(input_size, target_size)
Y = Y.reshape(-1, 1)
ds.setField('input', X)
ds.setField('target', Y)

print("building network")
# build network
hidden_size = 100
epochs = 200
continue_epochs = 10
validation_proportion = 0.20

nn = buildNetwork(input_size, hidden_size, target_size, bias=True)
bp = BackpropTrainer(nn, ds)

print("training...")
# train with cv
# bp.trainUntilConvergence(verbose=True, validationProportion=0.20, maxEpochs=1000, continueEpochs=10)

train_mse, validation_mse = bp.trainUntilConvergence(verbose=True, validationProportion=validation_proportion,
                                                     maxEpochs=epochs, continueEpochs=continue_epochs)

# pickle.dump(nn, open('model_val.pkl', 'wb'))
print("output")
for i, j in zip(train_mse, validation_mse):
    print("train_mse", i, "\tvalidation", j)
