#!/usr/bin/env python3
"""Module for concatenating 2D matrices."""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specified axis.

    Parameters:
    - mat1 (List[List[int/float]]): The first 2D matrix to concatenate.
    - mat2 (List[List[int/float]]): The second 2D matrix to concatenate.
    - axis (int, optional): The axis along which to concatenate the matrices.
        - 0 for vertical concatenation (default).
        - 1 for horizontal concatenation.

    Returns:
    - List[List[int/float]]: The resulting concatenated matrix as a new 2D list.
    - None: If concatenation is not possible due to mismatched dimensions.

    The function verifies the compatibility of `mat1` and `mat2` for concatenation
    based on the specified `axis`. For `axis=0`, the matrices must have the same number of columns. For `axis=1`, they must have the same number of rows.
    """
    # Check for dimension compatibility based on axis
    if len(mat1[0]) != len(mat2[0]) and axis == 0:
        return None
    if len(mat1) != len(mat2) and axis == 1:
        return None

    CAT = []  # Resultant matrix
    # Vertical concatenation
    if axis == 0:
        CAT.extend(mat1)  # Add all rows of mat1
        CAT.extend(mat2)  # Add all rows of mat2
    # Horizontal concatenation
    else:
        for i in range(len(mat1)):
            CAT.append(mat1[i] + mat2[i])  # Combine and add rows side by side

    return CAT
