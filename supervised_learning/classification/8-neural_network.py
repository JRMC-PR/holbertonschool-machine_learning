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
                W1: weights vector for the hidden layer
                b1: bias for the hidden layer
                A1: activated output for the hidden layer
                W2: weights vector for the output neuron
                b2: bias for the output neuron
                A2: activated output for the output neuron
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
        self.W1 = np.random.randn(nodes, nx)  # Weights for the hidden layer
        self.b1 = np.zeros((nodes, 1))  # Bias for the hidden layer
        self.A1 = 0  # Activated output for the hidden layer
        self.W2 = np.random.randn(1, nodes)  # Weights for the output neuron
        self.b2 = 0  # Bias for the output neuron
        self.A2 = 0  # Activated output for the output neuron
