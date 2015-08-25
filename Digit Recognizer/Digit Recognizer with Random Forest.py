__author__ = 'qiushi'

import numpy as np

train_data = np.genfromtxt("F:\ML\kaggle\Digit Recognizer\\train.csv", delimiter=',', skip_header=1, dtype=np.int)
test_data = np.genfromtxt("F:\ML\kaggle\Digit Recognizer\\test.csv", delimiter=',', skip_header=1, dtype=np.int)

X = train_data[:, 1:]
target = train_data[:, 0]

