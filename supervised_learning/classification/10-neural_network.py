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
