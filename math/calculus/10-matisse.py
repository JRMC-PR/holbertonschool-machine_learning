#!/usr/bin/env python3
"""This module calculates the derivative of a polynomial"""

def poly_derivative(poly):
    """This function calculates the derivative of a polynomial"""
    if not poly or not isinstance(poly, list):
        return None

    if len(poly) == 1:  # or poly == [0] * len(poly): if you want to check if all elements are zero
        return [0]

    # Initialize an empty list to hold the derivative coefficients
    derivative = []

    # Start from 1 because the derivative of the constant term (index 0) is 0
    for i in range(1, len(poly)):
        # Multiply the coefficient by its index, which is the power of x
        derivative.append(i * poly[i])

    # Handle the case where the derivative might still be zero
    if not derivative:  # This would be true if poly was something like [3, 0, 0, 0]
        return [0]

    return derivative
