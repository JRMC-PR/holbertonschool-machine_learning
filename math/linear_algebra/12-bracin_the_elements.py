#!/usr/bin/env python3

def np_elementwise(mat1, mat2):
    """
    Performs element-wise arithmetic operations on two numpy arrays.

    Parameters:
    - mat1 (numpy.ndarray): The first input array.
    - mat2 (numpy.ndarray): The second input array.

    Returns:
    - tuple: Contains the results of element-wise sum,
    difference, product, and quotient.
    """
    sum_result = mat1 + mat2
    difference_result = mat1 - mat2
    product_result = mat1 * mat2
    quotient_result = mat1 / mat2

    return (sum_result, difference_result, product_result, quotient_result)
