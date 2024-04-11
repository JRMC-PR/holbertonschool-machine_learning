#!/usr/bin/env python3
"""Module for performing matrix multiplication."""


def mat_mul(mat1, mat2):
    """
    Performs the matrix multiplication of two 2D lists (matrices).

    Parameters:
    - mat1 (List[List[int/float]]): The first matrix as a list of lists where inner lists represent rows, containing integers or floats.
    - mat2 (List[List[int/float]]): The second matrix as a list of lists where inner lists represent rows, containing integers or floats.

    Returns:
    - List[List[int/float]]: A new matrix (list of lists) representing the product of mat1 and mat2. If the number of columns in mat1 is not equal to the number of rows in mat2, returns None.

    Note:
    The function assumes that the input matrices are compatible for multiplication, i.e., the number of columns in the first matrix (mat1) is equal to the number of rows in the second matrix (mat2).
    """
    # if the number of columns of mat1 is not equal
    # to the number of rows of mat2
    if len(mat1[0]) != len(mat2):
        return None

    # create a list of lists with the number of rows
    # of mat1 and the number of columns of mat2
    # initialize all values to 0
    new_mat = [[0 for col in range(len(mat2[0]))] for row in range(len(mat1))]

    # iterate through rows of mat1
    for i in range(len(mat1)):
        # iterate through columns of mat2
        for j in range(len(mat2[0])):
            # iterate through columns of mat1
            for k in range(len(mat1[0])):
                # perform element-wise multiplication and sum
                new_mat[i][j] += mat1[i][k] * mat2[k][j]
    return new_mat
