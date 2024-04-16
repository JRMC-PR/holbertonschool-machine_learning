#!/usr/bin/env python3
"""Plot a line graph"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """Plot a line graph"""

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)  # Create an array for the x-axis values

    plt.plot(y, 'r-')  # Plot y as a solid red line
    plt.show()  # Display the plot
