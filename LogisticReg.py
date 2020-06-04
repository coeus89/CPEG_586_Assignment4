import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from Utils import Utils

class LogisticReg(object):
    beta = np.zeros((100,10))

    def sigmoid(self,X):
        return 1.0/(1.0 + np.exp(-np.dot(X, self.beta.T)))
    
    def gradientBeta(self, X, y):
        # (a - y) * x = dL/dBeta for gradient descent
        a = self.sigmoid(X)
        part1 = a - y.reshape(X.shape[0],1)
        grad = np.dot(part1.T, X)
        return grad

    def logLoss(self, X, y):
        a = self.sigmoid(X)
        loss = -(y * np.log(a) + (1 - y) * np.log(1 - a))
        return np.sum(loss)
    
    def trainLogisticRegUsingGradientDescent(self,X,y,num_iter,alpha = 0.01):
        loss = self.logLoss(X,y)
        for i in range(num_iter):
            self.beta = self.beta - (alpha * self.gradientBeta(X,y))
            loss = self.logLoss(X,y)
            if (i%10 == 0):
                print('iter = ' + str(i) + ' loss = ' + str(loss))
    
    def classifyData(self, X): #0 or 1
        a = self.sigmoid(X) # actual output
        decision = np.where(a >= 0.5, 1, 0)
        return decision
    
    def createAndTestLogisticReg(self, X, y):
        # train the logistic regression classifier
        utils = Utils()
        X1 = np.hstack((np.ones((1,X.shape[0])).T, X)) #add ones to the column data

        Y = y
        self.beta = np.zeros((1,X1.shape[1])) # (1,3) in this example
        y_predicted = self.classifyData(X1) # predictions by the trained model
        print("Mumber of correct predictions = ", str(np.sum(Y == y_predicted.reshape(Y.shape[0]))/len(X1)*100) + '%')
        #plot decision boundry
        self.plot_decision_boundary(lambda x: self.classifyData(x), X, y)
        plt.title("Logistic Regression")
        plt.show()

    def plot_decision_boundary(self, pred_func, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
        y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
        h = 0.01
        #generate a grid of points with distance h between them
        xx,yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
        # predict the function value for the whole grid
        x1 = np.c_[xx.ravel(),yy.ravel()]
        x1 = np.hstack((np.ones((1,x1.shape[0])).T, x1))
        Z = pred_func(x1) # will call classify data
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral)
        plt.scatter(X[:,0],X[:,1], c=y, cmap=plt.cm.Spectral)
