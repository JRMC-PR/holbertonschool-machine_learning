#!/usr/bin/env python3
"""Module for adding two lists element-wise."""


def add_arrays(arr1, arr2):
    """
    Adds two lists element-wise.

    This function takes two lists `arr1` and `arr2` of the same length and adds
    their corresponding elements. If the lists have different lengths, the function
    returns None to indicate that element-wise addition cannot be performed.

    Parameters:
    - arr1 (List[int/float]): The first list of integers or floats to be added.
    - arr2 (List[int/float]): The second list of integers or floats to be added.

    Returns:
    - List[int/float]: A new list where each element is the sum of the corresponding
    elements in `arr1` and `arr2`.
    - None: If `arr1` and `arr2` do not have the same length.

    Example:
    If arr1 = [1, 2, 3] and arr2 = [4, 5, 6], then the function returns [5, 7, 9].
    """
    # Check for equal length of the lists
    if len(arr1) != len(arr2):
        return None

    # Perform element-wise addition
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
