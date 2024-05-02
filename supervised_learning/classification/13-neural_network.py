#!/usr/bin/env python3
""" This module defines a newrual network
    class for binary classification"""
import numpy as np


class NeuralNetwork:
    """ This class defines a neural network with one hidden layer
        performing binary classification
    """
    def __init__(self, nx, nodes):
        """ Class constructor
            Args:
                nx: number of input features
                nodes: number of nodes found in the hidden layer
            Attributes:
                W1: weights vector of the hidden layer
                b1: bias of the hidden layer
                A1: activated output of the hidden layer
                W2: weights vector of the output neuron
                b2: bias of the output neuron
                A2: activated output of the output neuron
        """
        if isinstance(nx, int) is False:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if isinstance(nodes, int) is False:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # Initialize weights and biases
        self.__W1 = np.random.randn(nodes, nx)  # Weights of the hidden layer
        self.__b1 = np.zeros((nodes, 1))  # Bias of the hidden layer
        self.__A1 = 0  # Activated output of the hidden layer
        self.__W2 = np.random.randn(1, nodes)  # Weights of the output neuron
        self.__b2 = 0  # Bias of the output neuron
        self.__A2 = 0  # Activated output of the output neuron

    @property
    def W1(self):
        """ This method retrieves the weights vector of the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """ This method retrieves the bias of the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """ This method retrieves the activated output of the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """ This method retrieves the weights vector of the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """ This method retrieves the bias of the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """ This method retrieves the activated output of the output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """ This method calculates the forward propagation
            of the neural network
            Args:
                X: input data
            Returns:
                The activated output of the hidden layer and the
                activated output of the output neuron
        """
        # Calculate the hidden layer
        Z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        # Calculate the output neuron
        Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ This method calculates the cost of the model using
            logistic regression
            Args:
                Y: input data
                A: activated output of the neuron
            Returns:
                The cost
        """
        # number of examples
        m = Y.shape[1]
        # Compute the cost
        logprobs = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(logprobs)
        return cost

    def evaluate(self, X, Y):
        """ This method evaluates the neural network
            Args:
                X: input data
                Y: correct labels
            Returns:
                The activated output of the hidden layer and the
                activated output of the output neuron
        """
        # Perform forward propagation
        self.forward_prop(X)
        # Calculate the cost
        cost = self.cost(Y, self.A2)
        # Make the prediction
        prediction = np.where(self.A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ This method calculates one pass of gradient
            descent on the neural network
            Args:
                X: input data
                Y: correct labels
                A1: activated output of the hidden layer
                A2: activated output of the output neuron
                alpha: learning rate
        """
        # number of examoles in the input data
        m = Y.shape[1]
        # Calculate the gradient of the output neuron
        dz2 = A2 - Y
        dw2 = 1 / m * np.matmul(dz2, A1.T)
        db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
        # Calculate the gradient of the hidden layer
        dz1 = np.matmul(self.W2.T, dz2) * A1 * (1 - A1)
        dw1 = 1 / m * np.matmul(dz1, X.T)
        db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
        # Update the weights and biases
        # of the output neuron
        self.__W2 = self.W2 - alpha * dw2
        self.__b2 = self.b2 - alpha * db2
        # Update the weights and biases
        # of the hidden layer
        self.__W1 = self.W1 - alpha * dw1
        self.__b1 = self.b1 - alpha * db1
