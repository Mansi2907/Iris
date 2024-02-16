import Linear_regression
import train_regression1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

dataset =datasets.load_iris()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)

# weights  = np.load('Weight_Parameters.npy')
# bias = np.load('Bias_Parameters.npy')
                

model1 = Linear_regression.LinearRegression()
model1.fit(X_test[:,0], X_test[:,1])
predict = model1.predict(X_test[:,[0,1]])
# print('Weight is: ',weights)
# print('Bias is: ',bias)
mean_squared_error = model1.score(X_test[:,0],X_test[:,1])
print(mean_squared_error)