#!/usr/bin/env python3
""" concat 2D matrices """
import numpy as np


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenate 2D matrices """
    if len(mat1[0]) != len(mat2[0]) and axis == 0:
      return None
    return np.concatenate((mat1, mat2), axis=axis)
