#!/usr/bin/env python3
"""
This module contains a function to generate a line
graph representing the exponential decay of C-14.

The x-axis represents time in years and ranges from 0 to 28650.
The y-axis represents the fraction remaining and is logarithmically scaled.
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    This function generates a line graph representing
    the exponential decay of C-14.

    The x-axis represents time in years and ranges from 0 to 28650.
    The y-axis represents the fraction remaining and is logarithmically scaled.
    """
    # Generate an array of x values ranging from 0 to 28650 with a step of 5730
    x = np.arange(0, 28651, 5730)
    # Calculate the natural logarithm of 0.5
    r = np.log(0.5)
    # Define the time constant
    t = 5730
    # Calculate the y values based on the exponential decay formula
    y = np.exp((r / t) * x)

    # Create a new figure with a specific size
    plt.figure(figsize=(6.4, 4.8))

    # Plot y against x as a line graph
    plt.plot(x, y)
    # Set the y-axis to logarithmic scale
    plt.yscale('log')
    # Set the x-axis range from 0 to 28650
    plt.xlim(0, 28650)
    # Label the x-axis as 'Time (years)'
    plt.xlabel("Time (years)")
    # Label the y-axis as 'Fraction Remaining'
    plt.ylabel("Fraction Remaining")
    # Set the title of the plot as 'Exponential Decay of C-14'
    plt.title("Exponential Decay of C-14")
    # Display the plot
    plt.show()
