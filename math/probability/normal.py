#!/usr/bin/env python3
"""This module contains the normal distribution class
"""


class Normal:
    """This class represents a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """Normal class constructor
        Args:
            data (List): List of the data to be used to estimate
            the distribution
            mean (float): Mean of the distribution
            stddev (float): Standard deviation of the distribution
        """
        # Check if data is given
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # calculate the mean and standard deviation values
            self.mean = float(sum(data) / len(data))
            # Calculate the standard deviation
            self.stddev = (
                sum([(x - self.mean) ** 2 for x in data]) / len(data)) ** 0.5
        else:
            # If data is not given
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
