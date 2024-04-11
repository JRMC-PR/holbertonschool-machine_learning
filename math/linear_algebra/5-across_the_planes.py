#!/usr/bin/env python3
"""Module for adding two matrices element-wise."""


def add_matrices2D(mat1, mat2):
    """
    Adds two matrices element-wise.

    Checks if the matrices `mat1` and `mat2` have the same dimensions. If not, the function returns None.
    If they do, it proceeds to add corresponding elements from each matrix.

    Parameters:
    - mat1 (List[List[int/float]]): The first 2D matrix, a list of lists where each inner list represents a row.
    Elements can be of type int or float.
    - mat2 (List[List[int/float]]): The second 2D matrix, with the same requirements for format and element
    type as `mat1`.

    Returns:
    - List[List[int/float]]: A new 2D matrix representing the element-wise sum of `mat1` and `mat2`. Each element in the resulting matrix is the sum of the corresponding elements in `mat1` and `mat2`.
    - None: If `mat1` and `mat2` do not have the same dimensions.

    The function ensures that both matrices are of the same size and then computes the sum of each corresponding element. The result is a new matrix with the same dimensions as the input matrices.
    """
    # Check for matching dimensions
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    # Compute element-wise addition
    return [
        [mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))] for i in range(len(mat1))
    ]
