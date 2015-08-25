__author__ = 'qiushi'

# This is a Kaggle starter program
# Only for starting DM with python

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# train_data = np.genfromtxt("F:\ML\kaggle\Digit Recognizer\\train.csv", delimiter=',', skip_header=1, dtype=np.int)
# test_data = np.genfromtxt("F:\ML\kaggle\Digit Recognizer\\test.csv", delimiter=',', skip_header=1, dtype=np.int)
train_data = pd.read_csv("F:\ML\kaggle\Digit Recognizer\\train.csv")
test_data = pd.read_csv("F:\ML\kaggle\Digit Recognizer\\test.csv")

'''
some basic information of datasets
'''
train_data_row, train_data_col = train_data.shape
test_data_row, test_data_col = test_data.shape

#############################################################################################
# put data into X,y,CV or test
# fix dirty data
# pre-processing data like feature scaling and mean normalization

X = train_data.iloc[:200, 1:]
y = train_data.iloc[:200, 0]
classifiers = []

CV_x = np.array(test_data)[1:]

#############################################################################################
'''define function for main code'''


# transfer y into binary type
# if y==1, that means y is the same as label, and treat other 0 as different type label
# y should be numpy array type
def train_one_vs_all(y, labels):
    y_train = (y == labels).astype(int)
    return y_train


'''main code'''
for label in sorted(np.unique(y)):  # np.unique find the what is in y and pick one of them out
    print("fit parameter for label:" + str(label))
    print("create logistic regression")
    l = LogisticRegression()
    print("succeed")
    print("transfer y...")
    y_train = train_one_vs_all(y, label)
    print("succeed")
    print("fitting data...")
    l.fit(X, y_train)
    print("succeed")
    classifiers.append(l)

X = np.array(X)
predict_row = []
print("predicting...")
for test_x in CV_x:
    predict_col = []
    for classifier in classifiers:
        # each column == result for each label
        predict_col.append(classifier.predict(test_x))  # the predict results of one test data for each label
    predict_row.append(predict_col)  # collect the predict results as a row

print("printing results")
predict_list = np.array(predict_row)
#  return the index of the maximum number in each row
predict_results = predict_list.argmax(axis=1)  # the length of results should be the length of test_data
print(predict_results.shape)

#############################################################################################

