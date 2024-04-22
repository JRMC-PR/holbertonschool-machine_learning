#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def gradient():

    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    # Create a scatter plot with a colorbar
    scatter = plt.scatter(x, y, c=z)
    plt.colorbar(scatter).set_label('elevation (m)')

    # Set the labels and title
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Mountain Elevation')

    # Display the plot
    plt.show()
