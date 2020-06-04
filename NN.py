import matplotlib.pyplot as plt
import numpy as np
import datetime

class NN(object):
    # hdim: number of nodes in the hidden layer
    model = {}
    def createNN(self, X, y, hdim):
        # Initialize the parameters to random values. We need to learn these.
        np.random.seed((datetime.datetime.now()).microsecond)
        input_dim = 2
        output_dim = 2
        model = self.trainNN(X, y, input_dim, hdim, output_dim)
        return model
    
    def trainNN(self, X, y, input_dim, hdim, output_dim, epochs = 10000):
        lr = 0.01 # learning rate for gradient descent
        reg_lambda = 0.01 # regularization coefficient

        # initialize weights and biases
        W1 = np.random.randn(input_dim, hdim) / np.sqrt(input_dim)
        b1 = np.zeros((1,hdim))
        W2 = np.random.randn(hdim, output_dim) / np.sqrt(hdim)
        b2 = np.zeros((1, output_dim))

        model = {}
        for i in range(0, epochs):
            #forward propagation
            z1 = X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            # is this softmax??? i think so.
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            #back propagation
            delta3 = probs
            delta3[range(len(X)), y] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1,2))  # derivative of tanh = (1-a)^2
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += reg_lambda * W2
            dW1 += reg_lambda * W1

            # gradient descent update of weights and biases
            W1 += -lr * dW1
            b1 += -lr * db1
            W2 += -lr * dW2
            b2 += -lr * db2

            # store parameters in the model
            model = {'W1' : W1, 'b1' : b1, 'W2' : W2, 'b2' : b2}
            if i % 1000 == 0:
                print("Loss after iteration %i: %f" %(i, self.calculate_loss(model,X,y)))
        return model
    
    def predict_output(self, model, X):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propaga`tion
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) #softmax
        return np.argmax(probs, axis=1)
    
    def createAndTestNN(self, X, y, numHiddenNeurons):
        model = self.createNN(X,y, numHiddenNeurons)
        z = self.predict_output(model,X)
        accuracy = (z == y).mean()
        print("accuracy NN = " + str(accuracy))
        # plot the decision boundary
        self.plot_decision_boundary(lambda x: self.predict_output(model,x), X, y)
        plt.title("Decision Boundary for hidden layer size " + str(numHiddenNeurons))
        plt.show()

    def plot_decision_boundary(self, pred_func, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole gid
        Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral) 
    
    def calculate_loss(self, model, X, y):
        reg_lambda = 0.01 # regularization coefficient
        num_examples = len(X) # training set size
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) #softmax
        #calculate the loss
        correct_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs)
        # ddd regulatization term to loss
        data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1.0/num_examples * data_loss