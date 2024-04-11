#!/usr/bin/env python3
"""Module for transposing numpy arrays."""


def np_transpose(matrix):
    """
    Transposes a numpy array.

    Parameters:
    - matrix (numpy.ndarray): The numpy array to be transposed.

    Returns:
    - numpy.ndarray: A new numpy array which is the transpose of the input array.
    """
    return matrix.transpose()
