#!/usr/bin/env python3
"""Module for concatenating two arrays."""


def cat_arrays(arr1, arr2):
    """
    Concatenates two arrays (lists) into a single array (list).

    Parameters:
    - arr1 (List[int/float/any]): The first array to concatenate.
    Can contain elements of any type.
    - arr2 (List[int/float/any]): The second array to concatenate.
    Can contain elements of any type.

    Returns:
    - List[int/float/any]: A new array resulting from the concatenation of `arr1` and `arr2`.

    This function directly combines `arr1` and `arr2` using the `+` operator, effectively
    appending all elements of `arr2` to `arr1` and returning the new combined list.
    """
    CAT = arr1 + arr2
    return CAT
