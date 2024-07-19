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

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes.

        The Probability Mass Function (PMF) for the
        Poisson distribution calculates
        the probability of observing exactly k successes
        (events, arrivals, etc.)
        given the average rate (lambda) of successes in
        the same time period under
        consideration.

        Args:
            k: The number of successes for which to calculate
            the PMF. This is
            the specific value of the random variable for which
            the probability
            is calculated.

        Returns:
            The PMF value for observing exactly k successes,
            given the Poisson
            distribution's lambda (average rate of success).

        Note:
            - The PMF is defined only for non-negative integers.
            If a non-integer
            is provided, it is converted to an integer.
            - If k is negative, the function returns 0, as the
            PMF is not defined
            for negative numbers of successes.
        """
        # Define the base of the natural logarithm (e) to
        # approximate calculations.
        e = 2.7182818285

        # Ensure k is an integer, as the PMF is defined only
        # for integer values.
        if not isinstance(k, int):
            k = int(k)

        # The PMF is 0 for negative values of k, as negative
        # successes are not possible.
        if k < 0:
            return 0

        # Calculate k factorial (k!) as it's required for the
        # PMF formula.
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # Calculate and return the PMF using the formula
        # P(k; lambda) = (e^-lambda) * (lambda^k) / k!
        # This formula represents the probability of
        # observing exactly k successes.
        return ((e ** -self.lambtha) * (self.lambtha ** k)) / factorial
