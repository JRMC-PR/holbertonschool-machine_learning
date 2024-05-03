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
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        # Calculate the output neuron
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
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
        # number of examples in the input data
        m = Y.shape[1]
        # Calculate the gradient of the output neuron
        dz2 = A2 - Y
        dW2 = np.matmul(A1, dz2.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        # Calculate the gradient of the hidden layer
        dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
        dW1 = np.matmul(X, dz1.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        # Update the weights and biases of the output neuron
        self.__W2 = self.__W2 - alpha * dW2.T
        self.__b2 = self.__b2 - alpha * db2
        # Update the weights and biases of the hidden layer
        self.__W1 = self.__W1 - alpha * dW1.T
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ This method trains the neural network
            Args:
                X: input data
                Y: correct labels
                iterations: number of iterations to train over
                alpha: learning rate
            Returns:
                The evaluation of the training data after
                iterations of training have occurred
        """
        if isinstance(iterations, int) is False:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if isinstance(alpha, float) is False:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            # Perform forward propagation
            self.forward_prop(X)
            # Perform backpropagation
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        return self.evaluate(X, Y)
