#!/usr/bin/env python3
"""Module for transposing a 2D matrix."""


def matrix_transpose(matrix):
    """
    Return the transpose of a 2D matrix.

    The transpose of a matrix is obtained by swapping the rows and columns of the matrix.
    If the input matrix is of size m x n (m rows and n columns), the resulting transposed
    matrix will be of size n x m (n rows and m columns).

    Parameters:
    - matrix (List[List[int/float]]): A 2D list where inner lists represent rows of the matrix.
    The elements of the matrix can be of type int or float.

    Returns:
    - List[List[int/float]]: A new 2D list representing the transposed matrix. The original matrix is not modified.

    Example:
    If matrix = [[1, 2, 3], [4, 5, 6]], then the function returns [[1, 4], [2, 5], [3, 6]].
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
