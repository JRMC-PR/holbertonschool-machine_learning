#!/usr/bin/env python3
"""This moduel defines a class (Neuron) that defines a single neuron"""
import numpy as np


class Neuron:
    """This class defines a single neuron performing binary classification"""
    def __int__(self, nx):
        """This function initializes the Neuron instance"""
        if isinstance(nx, int) is False:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # Attributes
        self.W = np.random.randn(nx).reshape(1, nx)
        self.b = 0
        self.A = 0
