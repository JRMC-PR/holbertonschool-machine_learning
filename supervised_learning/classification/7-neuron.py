#!/usr/bin/env python3
"""This moduel defines a class (Neuron) that defines a single neuron"""
from matplotlib import pyplot as plt
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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        This function calculates one pass of gradient descent on the neuron
        Args:
            X (numpy.ndarray): contains the input data
            Y (numpy.ndarray): contains the correct labels
            A (numpy.ndarray): containing the activated output of the neuron
            alpha (float): learning rate
        """
        # get numberof examples in X
        m = Y.shape[1]
        # Calculate the gradient
        dz = A - Y
        # Derivative of the weight
        dw = np.matmul(X, dz.T) / m
        # Derivative of the bias
        db = np.sum(dz) / m
        # set the new weights and bias
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        This function trains the neuron

        Args:
            X (numpy.ndarray): contains the input data
            Y (numpy.ndarray): contains the correct labels
            iterations (int, optional): Defaults to 5000.
            alpha (float, optional): Learning rate. Defaults to 0.05.
        """
        # Validate iterations
        if isinstance(iterations, int) is False:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        # Validate alpha
        if isinstance(alpha, float) is False:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # Initialize cost list for graphing
        costs = []

        # Loop over the number of iterations + 1
        for i in range(iterations + 1):
            # Perform forward propagation
            self.forward_prop(X)
            # Calculate the cost
            cost = self.cost(Y, self.__A)
            #TODO: cherck the verbose and graph first before calculating the cost
            # If the current iteration is a multiple of
            # step or the last iteration
            if i % step == 0 or i == iterations:
                # If verbose is True, print the cost
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
                # If graph is True, append the cost to the
                # costs list for later graphing
                if graph is True:
                    costs.append(cost)
            # If not the last iteration, perform gradient
            # descent to update the weights and bias
            if i < iterations:
                self.gradient_descent(X, Y, self.__A, alpha)

        # If graph is True, plot the costs over the iterations
        if graph is True:
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel('iteration')  # Label the x-axis as 'iteration'
            plt.ylabel('cost')  # Label the y-axis as 'cost'
            plt.title('Training Cost')  # Title the plot as 'Training Cost'
            plt.show()  # Display the plot

        # Return the evaluation of the training data
        # after iterations of training
        return self.evaluate(X, Y)
