import Linear_regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


dataset =datasets.load_iris()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model1 = Linear_regression.LinearRegression()
model1.fit(X_train[:,0], X_train[:,2])
predict = model1.predict(X_train[:,[0,2]])
# print('Weight is: ',weights)
# print('Bias is: ',bias)
mean_squared_error = model1.score(X_train[:,0],X_train[:,2])