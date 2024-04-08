#!/usr/bin/env python3
"""adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """adds tow listby element """
    if len(arr1) != len(arr2):
        return None
    nl = []
    return [nl.append(arr1[i] + arr2[i]) for i in range(len(arr1))]
