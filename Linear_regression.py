import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        self.learning_rate = 0.2
        
    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # TODO: Initialize the weights and bias based on the shape of X and y.
        self.weights = 0
        self.bias = 0
        n = X.shape[0]
        steps = []
        loss =[]
        best_loss = float('inf')
        patience_counter = 0
        total_loss = 0 
        
        #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)
        for i in range(max_epochs):
            for j in range(0,len(X)+1,batch_size):
                
                end = j + batch_size
                x_hat = X[j:end]
                y_hat = y[j:end]
                #predicted value of y
                y_pred = np.dot(self.weights,x_hat)+self.bias
                #step loss
                error = y_hat-y_pred
                c_loss = np.sum((error)**2)/len(y_hat)
                loss.append(c_loss)
                steps.append(i+1)
                #calculate gradient
                gradients = np.dot(x_hat.T, error) / batch_size
                
                #update bias & weights
                self.weights -= self.learning_rate * gradients 
                self.bias -= self.learning_rate * np.mean(error)
                
                np.save('Weight_Parameters',self.weights)
                np.save('Bias_Parameters', self.bias)
                
                print("Step no: " , i+1)
                print("batch size:" , batch_size)
                print("LOSS is :" , c_loss)
#                 print("weight:",self.weights)
#                 print("Bias:",self.bias)
                
                self.weights  = np.load('Weight_Parameters.npy')
                self.bias = np.load('Bias_Parameters.npy')
                print('weight is: ', self.weights)
                print('bias is: ', self.bias)
                
                #batch loss
                batch_loss = np.mean((y_pred - y_hat) ** 2)
                total_loss += batch_loss
                avg_loss = total_loss / (n // batch_size)

                # Calculate validation loss
                y_val = self.predict(X)
                val_loss = np.mean((y_val - y) ** 2)

                # Check if validation loss has improved
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= patience:
                    break
                
        plt.plot(steps, loss)
        plt.style.use('fivethirtyeight')
        plt.figure(figsize = (8,6))
        plt.ylabel("Loss")
        plt.xlabel("Steps")        
                
        # TODO: Implement the training loop.

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
        
        pass

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        y_pred = self.predict(X)
        # print(y_pred.shape)
        # print(y.shape)
        error = y_pred-y
        
        mse = np.mean((error) ** 2)
        return mse
        pass