#!/usr/bin/env python3
"""This module calculates the sum of the
squares of the first n natural numbers"""


def summation_i_squared(n):
        """This function calculates the sum of the
        squares of the first n natural numbers"""
        if isinstance(n, (int, float)) and n < 1:
                return None

        if n == 1:
                return 1

        return n ** 2 + summation_i_squared(n - 1)
