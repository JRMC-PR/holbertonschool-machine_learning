#!/usr/bin/env python3
"""This module contains the minor function"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """Calculate the minor matrix of a given square matrix.

    Args:
        matrix (list of list of int/float): The matrix to calculate
        the minor for.

    Returns:
        list of list of int/float: The minor matrix of the given matrix.

    Raises:
        TypeError: If the input matrix is not a list of lists.
        ValueError: If the input matrix is not square or is empty.
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise ValueError("matrix must be a list of lists")
    # Determine if its a 0 x 0 matrix [[]]
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    # Determine if its a 1 x 1 matrix
    if len(matrix) == 1:
        return [[1]]

    # Validate the matrix is square and non-empty
    if len(matrix) == 0 or not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Calculate the minor matrix
    minor_matrix = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix)):
            # Create a sub-matrix excluding the current row and column
            sub_matrix = [
                row[:j] + row[j + 1:] for k, row in enumerate(matrix) if k != i
            ]
            # Calculate the determinant of the sub-matrix
            if len(sub_matrix) > 1:  # Ensure the sub-matrix is at least 2x2
                minor_row.append(determinant(sub_matrix))
            else:
                # For a 1x1 sub-matrix, the determinant is the element itself
                minor_row.append(sub_matrix[0][0])
        minor_matrix.append(minor_row)

    return minor_matrix
