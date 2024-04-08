#!/usr/bin/env python3
"""definig a function that returns the shape of a matrix"""
def matrix_shape(matrix):
    """Find the shapeof the matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
