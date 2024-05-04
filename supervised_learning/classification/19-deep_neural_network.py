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
