#!/usr/bin/env python3
"""This module create a one hot matrix
from a numpy array"""
import numpy as np


def one_hot_encode(Y, classes):
    """This function create a one hot matix from a numpy array
        Args:
            Y: numpy.ndarray - shape(m,) contains the numeric class labels
            classes: int - the maximum number of classes found in Y
        Return: A one hot encoding of Y with shape (classes, m)
                or None if failed
    """
    # Check if Y is a numpy array and classes is an integer
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    # Check if Y is empty
    if len(Y) == 0:
        return None
    # Check if classes <= max value in Y
    if classes <= np.max(Y):
        return None
    try:
        # Create a zero matrix of shape (classes, Y.shape[0])
        one_hot = np.zeros((classes, Y.shape[0]))
        # Set the appropriate elements to 1
        one_hot[Y, np.arange(Y.shape[0])] = 1
        # Return the resulting one-hot matrix
        return one_hot
    except Exception:
        # If any error occurs, return None
        return None
