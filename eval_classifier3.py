import LDA
from sklearn import datasets
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import math




dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1, random_state = 0)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model = LDA.LinearDiscriminantAnalysis()
model.fit(X_test, y_test)



