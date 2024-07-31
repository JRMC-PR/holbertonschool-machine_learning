#!/usr/bin/env python3
"""This modlue initializes a cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """This function initializes cluster centroids for K-means
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset that will be
        used for K-means clustering
            - n is the number of data points
            - d is the number of dimensions for each data point
        k: positive integer containing the number of clusters
        Returns:
            numpy.ndarray of shape (k, d) containing the initialized
            centroids for each cluster, or None on failure
    """
    # Check for input data
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    # Extract the shape of the dataset
    n, d = X.shape
    # XCheck for inpuit data k
    if not isinstance(k, int) or k <= 0 or k > n:
        return None

    # Generate k initial centroids by sampling from a uniform
    # distribution
    # The uniform distribution is defined by the minimum and
    # maximum values
    # of the dataset X along each dimension

    # np.min(X, axis=0) computes the minimum value for each
    # feature (column) in X
    # np.max(X, axis=0) computes the maximum value for each
    # feature (column) in X
    # These min and max values define the range for the uniform
    # distribution

    # np.random.uniform generates random numbers from a uniform
    # distribution
    # with the specified low and high bounds for each feature
    # size=(k, d) specifies that we want to generate k samples,
    # each with d features
    centroids = np.random.uniform(
        low=np.min(X, axis=0), high=np.max(X, axis=0), size=(k, d)
    )
    return centroids
