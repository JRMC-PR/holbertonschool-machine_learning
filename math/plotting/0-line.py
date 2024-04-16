#!/usr/bin/env python3
"""Plotting module
    """
import numpy as np
import matplotlib.pyplot as plt

def line():
    """Plot y as a line graph
    """

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, 'r-')  # Plot y as a solid red line
    plt.xlim([0, 10])  # Set the x-axis range from 0 to 10

    plt.show()  # Display the plot
