#!/usr/bin/env python3
"""
This module contains a function to generate a line graph
representing the exponential decay of C-14 and Ra-226.

The x-axis represents time in years and ranges from 0 to 20000.
The y-axis represents the fraction remaining and ranges from 0 to 1.
"""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    This function generates a line graph representing
    the exponential decay of C-14 and Ra-226.

    The x-axis represents time in years and ranges from 0 to 20000.
    The y-axis represents the fraction remaining and ranges from 0 to 1.
    """
    # Generate an array of x values ranging from 0 to 21000 with a step of 1000
    x = np.arange(0, 21000, 1000)
    # Calculate the natural logarithm of 0.5
    r = np.log(0.5)
    # Define the time constants for C-14 and Ra-226
    t1 = 5730
    t2 = 1600
    # Calculate the y values for C-14 and Ra-226 based
    # on the exponential decay formula
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)

    # Create a new figure with a specific size
    plt.figure(figsize=(6.4, 4.8))

    # Label the x-axis as 'Time (years)'
    plt.xlabel("Time (years)")
    # Label the y-axis as 'Fraction Remaining'
    plt.ylabel("Fraction Remaining")
    # Set the title of the plot as 'Exponential Decay of Radioactive Elements'
    plt.title("Exponential Decay of Radioactive Elements")
    # Set the x-axis range from 0 to 20000
    plt.xlim(0, 20000)
    # Set the y-axis range from 0 to 1
    plt.ylim(0, 1)
    # Plot y1 against x as a red dashed line and label it as 'C-14'
    plt.plot(x, y1, 'r--', label='C-14')
    # Plot y2 against x as a green solid line and label it as 'Ra-226'
    plt.plot(x, y2, 'g', label='Ra-226')
    # Display the legend
    plt.legend()
    # Display the plot
    plt.show()
