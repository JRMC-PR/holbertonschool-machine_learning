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

    def z_score(self, x):
        """Calculates the z-score of a given x-value
        Args:
            x (float): The x-value
        Returns:
            The z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score
        Args:
            z (float): The z-score
        Returns:
            The x-value of z
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """This method calculates the value of the PDF for a
        given x-value in normal distribution
        Args:
            x (float): The x-value
        Returns:
            The PDF value for x
        """
        e = 2.7182818285
        pi = 3.1415926536
        pdf_val = 0
        # Calculate the PDF value for normal distribution
        pdf_val = (1 / (self.stddev * (2 * pi) ** 0.5)) * \
            e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
        return pdf_val
