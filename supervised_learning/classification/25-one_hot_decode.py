#!?usr/bin/env python3
"""This module defines a function that converts a one-hot
matrix into a vector of labels"""
import numpy as np


def one_hot_decode(one_hot):
    """ Converts a one-hot matrix into a vector of labels
        Args:
            one_hot: numpy.ndarray - shape(classes, m)
                contains the one-hot matrix
        Return: A numpy.ndarray - shape(m,) containing the
            numeric labels for each example, or None on failure
    """
    # Check if one_hot is a numpy array
    if not isinstance(one_hot, np.ndarray):
        return None
    try:
        # Use argmax to find the indices of the maximum values (1s)
        # along the axis representing the classes
        labels = np.argmax(one_hot, axis=0)
        # Return the resulting label vector
        return labels
    except Exception:
        # If any error occurs, return None
        return None
