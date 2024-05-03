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
                layers: list representing the number of nodes in each layer
            Attributes:
                L: The number of layers in the neural network
                cache: dictionary to hold all intermediary values of the network
                weights: dictionary to hold all weights and biases of the network
        """
        # Check if nx is an integer, if not raise a TypeError
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        # Check if nx is a positive integer, if not raise a ValueError
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # Check if layers is a list of positive integers, if not raise a TypeError
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        # Check if all elements in layers are positive integers, if not raise a TypeError
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        # Initialize number of layers
        self.L = len(layers)
        # Initialize cache as an empty dictionary
        self.cache = {}
        # Initialize weights as an empty dictionary
        self.weights = {}
        for i in range(self.L):
            # Initialize weights using He et al. method
            # If it's the first layer, the weights are based on the number of input features nx
            if i == 0:
                self.weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], nx) * np.sqrt(2/nx)
            # For subsequent layers, the weights are based on the number of nodes in the previous layer
            else:
                self.weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], layers[i - 1]) * \
                    np.sqrt(2/layers[i - 1])
            # Initialize biases to 0's for each layer
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
