#!/usr/bin/env python3
""" This module defines a deep neural network
class for binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """ This class defines a deep neural network
        performing binary classification
    """
    def __init__(self, nx, layers):
        """ Class constructor
            Args:
                nx: number of input features
                layers: list representing the number of
                nodes in each layer
            Attributes:
                L: The number of layers in the neural network
                cache: dictionary to hold all intermediary values
                of the network
                weights: dictionary to hold all weights and biases
                of the network
        """
        # Check if nx is an integer, if not raise a TypeError
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        # Check if nx is a positive integer, if not raise a ValueError
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # Check if layers is a list of positive integers,
        # if not raise a TypeError
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        # Check if all elements in layers are positive integers,
        # if not raise a TypeError
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        # Initialize number of layers
        self.__L = len(layers)
        # Initialize cache as an empty dictionary
        self.__cache = {}
        # Initialize weights as an empty dictionary
        self.__weights = {}
        for i in range(self.L):
            # Initialize weights using He et al. method
            # If it's the first layer, the weights are based
            # on the number of input features nx
            if i == 0:
                self.weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], nx) * np.sqrt(2/nx)
            # Fr subsequent layers, the weights are based on the number
            # of nodes in the previous layer
            else:
                self.weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], layers[i - 1]) * \
                    np.sqrt(2/layers[i - 1])
            # Initialize biases to 0's fr each layer
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ This method retrieves the number of layers"""
        return self.__L

    @property
    def cache(self):
        """ This method retrieves the intermediary values"""
        return self.__cache

    @property
    def weights(self):
        """ This method retrieves the weights and biases"""
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network
            Args:
                X: numpy.ndarray with shape (nx, m) that contains
                the input data
            Returns:
                The output of the neural network and the cache, respectively
        """
        # Save the input data to the cache dictionary
        self.cache['A0'] = X
        # Loop over all layers
        for i in range(self.__L):
            # Calculate the net input fr the current layer
            W_key = 'W' + str(i + 1)
            b_key = 'b' + str(i + 1)
            A_key = 'A' + str(i)

            Z = np.dot(self.weights[W_key], self.__cache[A_key])
            Z += self.weights[b_key]
            # Apply the sigmoid activation function
            self.__cache['A' + str(i + 1)] = 1 / (1 + np.exp(-Z))
        # Return the output of the neural network and the cache
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression
            Args:
                Y: numpy.ndarray with shape (nx, m) that contains
                the input data
                A: numpy.ndarray with shape (1, m) that contains
                the correct Activation output of the network
            Returns:
                The cost
        """
        # Number of examples
        m = Y.shape[1]
        # Compute the cost
        logprob = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(logprob)
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network's predictions
            Args:
                X: numpy.ndarray with shape (nx, m) that contains
                the input data
                Y: numpy.ndarray with shape (1, m) that contains
                the correct labels
            Returns:
                The predicted labels for X, and the cost of the network
        """
        # Preform forward propragation
        A, _ = self.forward_prop(X)
        # Calculate the cost
        cost = self.cost(Y, A)
        # Apply the sigmoid activation function
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient
            descent on the neural network
            Args:
                X: numpy.ndarray with shape (nx, m) that contains
                the input data
                Y: numpy.ndarray with shape (1, m) that contains
                the correct labels
                cache: dictionary containing all intermediary values
                of the network
                alpha: learning rate
        """
        # Number of examples in input data
        m = Y.shape[1]
        # Calculate the gradients of the output data
        dZ = cache['A' + str(self.L)] - Y
        for i in range(self.L, 0, -1):
            # Get the cached activations
            A_prev = cache['A' + str(i - 1)]
            # Calculate the derivatives of the weights and biases
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            # Calculate the derivative of the cost
            # with respect to the activation
            dZ_step1 = np.dot(self.weights['W' + str(i)].T, dZ)
            dZ = dZ_step1 * (A_prev * (1 - A_prev))
            # Update the weights and biases
            self.weights['W' + str(i)] -= alpha * dW
            self.weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the deep neural network
            Args:
                X: numpy.ndarray with shape (nx, m)
                that contains the input data
                Y: numpy.ndarray with shape (1, m)
                that contains the correct labels
                iterations: number of iterations to train over
                alpha: learning rate
            Returns:
                The evaluation of the training data after
                iterations of training
        """
        # Check the types and values of iterations and alpha
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Train the network
        for _ in range(iterations):
            # Perform forward propagation and calculate the cost
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
        # Return the evaluation of the training data
        return self.evaluate(X, Y)
