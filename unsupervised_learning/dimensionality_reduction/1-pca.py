#!/usr/bin/env python3
"""This modlue contains the function pca(X, ndim)"""
import numpy as np


def pca(X, ndim):
    """This function performs PCA on a dataset
    Args:
        X: numpy.ndarray of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions in each point
        ndim: the new dimensionality of the transformed X
        Returns:
        T: numpy.ndarray of shape (n, ndim) containing the transformed version
        of X
        """
    # # Ensure all dimetions have a mean between all data points
    # X = X - np.mean(X, axis=0)

    # # Compute the SVD:
    # U, S, Vt = np.linalg.svd(X)

    # # Compute the cumulative sum of the explained variance ratio
    # tr = np.matmul(U[..., :ndim], np.diag(S[..., :ndim]))

    # return tr

    # Ensure all dimensions have a mean between all data points
    X = X - np.mean(X, axis=0)

    # Compute the SVD:
    U, S, Vt = np.linalg.svd(X)

    # Select the top `ndim` components
    U_reduced = U[:, :ndim]
    S_reduced = np.diag(S[:ndim])

    # Compute the transformed data
    T = np.matmul(U_reduced, S_reduced)

    return T
