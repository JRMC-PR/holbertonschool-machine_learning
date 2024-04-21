#!/usr/bin/env python3
"""
This module contains a function that calculates the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Parameters:
    poly (list): A list of coefficients representing a
    polynomial. The index of each coefficient
                 corresponds to the power of its term.

    Returns:
    list: A list of coefficients representing the derivative
    of the input polynomial.
          Returns None if the input is not a list or is an
          empty list.
          Returns [0] if the input polynomial has a degree of 0
          (i.e., it is a constant).
    """

    # Check if poly is a list and is not empty
    if not poly or not isinstance(poly, list):
        return None

    # If the polynomial is a constant (degree 0), its derivative is 0
    if len(poly) == 1:
        return [0]

    # Initialize an empty list to hold the derivative coefficients
    derivative = []

    # Start from 1 because the derivative of the constant term (index 0) is 0
    for i in range(1, len(poly)):
        # Multiply the coefficient by its index, which is the power of x
        derivative.append(i * poly[i])

    # Handle the case where the derivative might still be zero
    # This would be true if poly was something like [3, 0, 0, 0]
    if not derivative:
        return [0]

    return derivative
