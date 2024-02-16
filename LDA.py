from sklearn import datasets
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from mlxtend.plotting import plot_decision_regions

class LinearDiscriminantAnalysis:
    def __init__(self):
        self.Class_label = None
        self.class_means = None
        self.shared_covariance = None
        self.Prior_prob = None

    def fit(self, X, y):
        self.Class_label = np.unique(y)
        self.class_means = np.array([np.mean(X[y == cls], axis=0) for cls in self.Class_label])
        self.shared_covariance = self.calculate_shared_covariance(X, y)
        self.Prior_prob = np.array([np.mean(y == cls) for cls in self.Class_label])

    def predict(self, X):
        y_pred = []
        for i in X:
            discriminants = [self.calculate_discriminant(i, cls) for cls in self.Class_label]
            predicted_class = self.Class_label[np.argmax(discriminants)]
            y_pred.append(predicted_class)
        return np.array(y_pred)

    def calculate_shared_covariance(self, X, y):
        total_samples = len(X)
        covariances = [np.cov(X[y == cls], rowvar=False) for cls in self.Class_label]
        return np.sum(covariances, axis=0) / total_samples

    def calculate_discriminant(self, x, cls):
        class_mean = self.class_means[self.Class_label == cls][0]
        inv_covariance = np.linalg.inv(self.shared_covariance)
        return np.dot(np.dot(x - class_mean, inv_covariance), (x - class_mean).T) - 2 * np.log(self.Prior_prob[self.Class_label == cls])

