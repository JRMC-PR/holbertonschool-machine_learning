#!/usr/bin/env python3
"""This module contains the function for task 1
that calculates the correlation matrix of a data set"""
import numpy as np


def correlation(C):
    """This function calculates the correlation matrix of a data set
    Args:
        C: numpy.ndarray - shape (d, d) containing the covariance matrix
    Returns:
        correlation: numpy.ndarray - shape (d, d) containing the
        correlation matrix
    """
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        raise TypeError("C must be a 2D numpy.ndarray")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix")
    if np.any(np.isnan(C)):
        raise ValueError("C contains NaN")
    if np.any(np.isinf(C)):
        raise ValueError("C contains infinity")
    # if np.any(np.diag(C) != 1):
    #     raise ValueError("C must be a valid covariance matrix")

    # Calculate the correlation matrix
    d = C.shape[0]
    D = np.diag(1 / np.sqrt(np.diag(C)))
    correlation = np.dot(np.dot(D, C), D)

    return correlation
