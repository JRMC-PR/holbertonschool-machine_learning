#!/usr/bin/env python3
""" Concatenates two arrays """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specified axis.

    Parameters:
    - mat1 (numpy.ndarray): The first matrix.
    - mat2 (numpy.ndarray): The second matrix.
    - axis (int, optional): The axis along which to concatenate. Default is 0.

    Returns:
    - numpy.ndarray: The concatenated matrix.
    """
    return np.concatenate((mat1, mat2), axis=axis)
