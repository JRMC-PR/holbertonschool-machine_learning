#!/usr/bin/env python3
"""This modlue contains the poisson distribution class
"""


class Poisson:
    """This class represents a poisson distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """Initialized the Poisson object with the given lambtha value
        Args:
            data: list of the data to be used to estimate the distribution
            lambtha: expected number of occurences in a given time frame
        """
        # Check if data is given
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # calculate the lambtha value
            self.lambtha = float(sum(data) / len(data))
        else:
            # If data is not given
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
