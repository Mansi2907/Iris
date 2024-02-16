from sklearn import datasets
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from mlxtend.plotting import plot_decision_regions


class LogisticRegression():
    
    def __init__(self,learning_rate = 0.01, max_epochs = 100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        
    def fit(self,X,y, learning_rate = 0.001, max_epochs = 100):
        
        self.num_features = X.shape
        self.weights = np.zeros(self.num_features)
        self.max_epochs = max_epochs
        self.bias = 0
        self.X = X
        self.y = y
        
        for i in range(max_epochs):

            num_samples, num_features = X.shape
            self.weights = np.zeros(num_features)
            self.bias = 0

            for _ in range(self.max_epochs):
                linear_output = np.dot(X, self.weights) + self.bias
                y_pred = 1 / (1 + np.exp(-linear_output))
                dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
                db = (1 / num_samples) * np.sum(y_pred - y)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
           
            
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = 1 / (1 + np.exp(-z))
        return np.round(y_pred)
    
    def plot_decision_boundary(self, X, y):
        plot_decision_regions(X, y, clf=self)
        plt.xlabel('Petal Length')
        plt.ylabel('Petal Width')
        plt.title('Logistic Regression')
        plt.show()
        
    def accuracy(self, y_n, y_p):
        score = np.sum(y_n==y_p) / len(y_n)
        return score 
        

