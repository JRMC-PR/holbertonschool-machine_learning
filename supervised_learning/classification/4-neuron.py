#!/usr/bin/env python3
"""This moduel defines a class (Neuron) that defines a single neuron"""
import numpy as np


class Neuron:
    """
    This class defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        This function initializes the Neuron instance

        Args:
            nx (Int): Number of input feaatures
        """
        if isinstance(nx, int) is False:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # Attributes
        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter for the weights vector

        Returns:
            __W: Contains the weights
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for the bias

        Returns:
            __b: Contains the bias
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for the activated output

        Returns:
            __A: Contains the activated output
        """
        return self.__A

    def forward_prop(self, X):
        """
        This function calculates the forward propagation of the neuron
        Args:
            X (numpy.ndarray): contains input data_

        Returns:
            __A: Contains the activated output
        """        """"""
        # Compute the wheighhted sum of the input values
        w_sum = np.matmul(self.__W, X) + self.__b
        # Sigmoid activation function
        self.__A = 1 / (1 + np.exp(-w_sum))
        return self.__A

    def cost(self, Y, A):
        """
        This function calculates the cost of
        the model using logistic regression
        Args:
            Y (numpy.ndarray): contains the correct labels for the input data
            A (numpy.ndarray): containing the activated output of the neuron

        Returns:
            float: representing the cost of the model
        """
        # number of examples
        m = Y.shape[1]
        # Compute the cost
        logprobs = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(logprobs)
        return cost

    def evaluate(self, X, Y):
        """
        This method evaluates the neuron's predictions
        Args:
            X (numpy.ndarray): contains the input data
            Y (numpy.ndarray): contains the correct labels
        Returns:
            tuple: containing the neuron's prediction
            and the cost of the network
        """
        # Perform forward propagation
        self.forward_prop(X)
        # Calculate the cost
        cost = self.cost(Y, self.__A)
        # Make the prediction
        prediction = np.where(self.__A >= 0.5, 1, 0)
        return prediction, cost
