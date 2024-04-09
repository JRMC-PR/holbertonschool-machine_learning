#!/usr/bin/env python3
""" matrix multiplication """


def mat_mul(mat1, mat2):
    """_summary_
    Takes in two matrices and returns the multiplication of the two matrices

    Args:
        mat1 (int/floats): Holds the matinx list
        mat2 (int/floats): Holds the matrix list
    """ """"""
    # if the number of columns of mat1 is not equal
    # to the number of rows of mat2
    if len(mat1[0]) != len(mat2):
        return None

    # create a list of lists with the number of rows
    # of mat1 and the number of columns of mat2
    # initialize all values to 0
    new_mat = [[0 for col in range(len(mat2[0]))] for row in range(len(mat1))]

    # iterate through the rows of mat1
    for i in range(len(mat1)):
        # iterate through the columns of mat2
        for j in range(len(mat2[0])):
            # iterate through the columns of mat1
            for k in range(len(mat1[0])):
                # multiply the corresponding values of mat1 and mat2
                new_mat[i][j] += mat1[i][k] * mat2[k][j]
    return new_mat
