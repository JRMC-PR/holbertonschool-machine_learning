#!/usr/bin/env python3
""" concat 2D matrices """


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenate 2D matrices """
    # if the number of columns are not equal
    if len(mat1[0]) != len(mat2[0]) and axis == 0:
      return None
    # if the number of rows are not equal
    if len(mat1) != len(mat2) and axis == 1:
        return None

    CAT = []
    # concatenate along columns
    if axis == 0:
      for i in mat1:
        CAT.append(i.copy())
      for j in mat2:
        CAT.append(j.copy())
    # concatenate along rows
    else:
      for i in range(len(mat1)):
          CAT.append(mat1[i] + mat2[i])
    return CAT
