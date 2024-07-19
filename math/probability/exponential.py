#!/usr/bin/env python3
"""This module contains the Exponential class"""


class Exponential:
    """This class represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Exponential class constructor
        Args:
            data (List): List of the data to be used to estimate
            the distribution
            lambtha (float): Expected number of occurences in a
            given time frame
        """
        # Check if data is given
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # calculate the lambtha value
            self.lambtha = float(sum(data) / len(data))
            # Inverse the lambtha value for the exponential distribution
            self.lambtha = 1 / self.lambtha
        else:
            # If data is not given
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
