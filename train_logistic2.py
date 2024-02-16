import Logistic_Regression
from sklearn import datasets
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from mlxtend.plotting import plot_decision_regions

dataset = datasets.load_iris()
X = dataset.data[:,[0,1]]
y = dataset.target
# print(X.shape)
# print(y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1, random_state = 0)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model = Logistic_Regression.LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.plot_decision_boundary(X_train, y_train)
Accuracy = model.accuracy(y_test,y_pred)
print("Accuracy is: ", Accuracy)