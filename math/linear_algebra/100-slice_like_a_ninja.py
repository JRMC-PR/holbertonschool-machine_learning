#!/usr/bin/env python3
"""slicing module for numpy arrays
    """
import numpy as np


def np_slice(matrix, axes={}):
    """
    Slices a numpy.ndarray along specified axes.

    Parameters:
    - matrix (numpy.ndarray): The input array to slice.
    - axes (dict): A dictionary where each key is an axis to slice along,
        and the value is a tuple representing the slice on that axis.

    Returns:
    - numpy.ndarray: The sliced array.
    """
    # Determine the number of axes (dimensions) in the input matrix
    num_axes = matrix.ndim

    # Prepare a list of slices, initially slicing entirely along each axis
    slices = [slice(None)] * num_axes

    # Update the slices based on the axes dictionary
    for axis, slice_range in axes.items():
        slices[axis] = slice(*slice_range)

    # Apply the prepared slices to the matrix and return the result
    return matrix[tuple(slices)]
