import LDA
from sklearn import datasets
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from mlxtend.plotting import plot_decision_regions



dataset = datasets.load_iris()
X = dataset.data[:,[2,3]]
y = dataset.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1, random_state = 0)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model = LDA.LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plot_decision_regions(X_train, y_train, clf = model, legend = 2)

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Linear Discriminant Analysis')
plt.show()