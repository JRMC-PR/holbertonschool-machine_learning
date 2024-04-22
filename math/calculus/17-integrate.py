#!/usr/bin/env python3
"""
This module contains a function that calculates the integral of a polynomial.
"""


def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial.

    Parameters:
    poly (list): A list of coefficients representing a
    polynomial. The index of each coefficient
                 corresponds to the power of its term.
    C (int): The integration constant.

    Returns:
    list: A list of coefficients representing the integral
    of the input polynomial.
          Returns None if the input is not a list, is an empty l
          ist, or if C is not an integer.
    """

    # Check if poly is a list and is not empty, and if C is an integer
    if not poly or not isinstance(poly, list) or not isinstance(C, int):
        return None

    # Initialize a list to hold the integral coefficients, starting
    # with the integration constant
    integral = [C]

    # For each coefficient in the polynomial...
    for i in range(len(poly)):
        # Divide the coefficient by its index + 1 (which is the
        # power of x after integration),
        # and round to 2 decimal places
        coef = poly[i] / (i + 1)

        # If the coefficient is a whole number, represent it as an integer
        if coef.is_integer():
            coef = int(coef)

        # Append the coefficient to the list of integral coefficients
        integral.append(coef)

    # Remove trailing zeros
    while integral and integral[-1] == 0:
        integral.pop()

    return integral
