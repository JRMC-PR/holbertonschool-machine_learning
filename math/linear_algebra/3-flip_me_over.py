#!/usr/bin/env python3
"""Transpose a matrix"""


def matrix_transpose(matrix):
    """Return the transpose of a 2D matrix"""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
