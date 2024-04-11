#!/usr/bin/env python3
"""Module for determining the shape of matrices."""


def matrix_shape(matrix):
    """
    Determines the shape of a matrix represented as a nested list.

    This function recursively explores the nested list structure (matrix)
    to find the size of each dimension, effectively determining the matrix's shape.
    It supports matrices of any dimensionality, not just 2D matrices, by traversing through
    nested lists until non-list elements are found.

    Parameters:
    - matrix (List, nested): A nested list where each level of nesting represents a dimension
    of the matrix. The innermost nested lists contain the matrix elements, which can be of any type.

    Returns:
    - List[int]: A list of integers where each integer represents the size of the matrix along that
    dimension. The first integer corresponds to the outermost list (highest dimension), and the last
    integer corresponds to the innermost list (lowest dimension).

    Example:
    - For a 2D matrix like [[1, 2], [3, 4]], it returns [2, 2].
    - For a 3D matrix like [[[1], [2]], [[3], [4]]], it returns [2, 2, 1].
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]  # Dive into the next dimension
    return shape
