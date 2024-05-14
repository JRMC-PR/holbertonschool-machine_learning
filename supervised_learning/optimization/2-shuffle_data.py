#!/usr/bin/env python3
"""This module contains the function for shuffling data
"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way.

    Args:
        X (numpy.ndarray): The first matrix of shape (m, nx) to shuffle.
        Y (numpy.ndarray): The second matrix of shape (m, ny) to shuffle.

    Returns:
        numpy.ndarray: The shuffled X and Y matrices.
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
