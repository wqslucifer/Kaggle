__author__ = 'qiushi'

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

X = train_data.iloc[:, 1:]
y = train_data.iloc[:, 0]
classifiers = []

CV_x = np.array(train_data.iloc[30001:30021, 1:])
CV_y = np.array(train_data.iloc[30001:30021, 0])
#############################################################################################
'''define function for main code'''

rf = RandomForestClassifier(n_estimators=100)
print("fitting data")
rf.fit(X, y)

predict = rf.predict(test_data)
print("completed")
# accuracy = sum((predict == CV_y).astype(int)) / CV_y.shape[0]
# print("CV accuracy is:" + accuracy)


print("saving the result")
pd.DataFrame({"ImageId": range(1, len(predict) + 1), "Label": predict}).to_csv(
    'F:\ML\kaggle\Digit Recognizer\submit.csv', index=False, header=True)

#############################################################################################
# END
# first submit with 0.96614
# NO.346
