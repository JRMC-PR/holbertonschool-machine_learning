#!/usr/bin/env python3
"""
This script generates a figure with five subplots using matplotlib and numpy.
"""

import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    This function generates and displays a figure with five subplots.
    """

    # Generate cubic values for the first plot
    y0 = np.arange(0, 11) ** 3

    # Generate data for the second plot
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    # Generate data for the third plot
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    # Generate data for the fourth plot
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    # Generate data for the fifth plot
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # Create a figure
    fig = plt.figure(figsize=(5, 5))
    fig.suptitle('All in One', fontsize='x-small')

    # Plot 1: y = x^3
    plt.subplot(3, 2, 1)
    plt.plot(y0, 'r')
    plt.xticks(range(0, 11, 2))
    plt.yticks(range(0, 1100, 500))

    # Plot 2: Men's Height vs Weight
    plt.subplot(3, 2, 2)
    plt.scatter(x1, y1, c='magenta')
    plt.xlabel('Height (in)', fontsize='x-small')
    plt.ylabel('Weight (lbs)', fontsize='x-small')
    plt.title('Men\'s Height vs Weight', fontsize='x-small')

    # Plot 3: Exponential Decay of C-14
    plt.subplot(3, 2, 3)
    plt.plot(x2, y2)
    plt.yscale('log')
    plt.xticks(range(0, 30000, 10000))
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.title('Exponential Decay of C-14', fontsize='x-small')

    # Plot 4: Exponential Decay of Radioactive Elements
    plt.subplot(3, 2, 4)
    plt.plot(x3, y31, 'r--', label='C-14')
    plt.plot(x3, y32, 'g-', label='Ra-226')
    plt.xticks(np.arange(0, 21000, step=5000))
    plt.yticks(np.arange(0, 1.2, step=0.5))
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.title('Exponential Decay of Radioactive Elements', fontsize='x-small')
    plt.legend(fontsize='x-small')

    # Plot 5: Student Grades
    plt.subplot(3, 2, (5, 6))
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.xticks(range(0, 101, 10))
    plt.xlabel('Grades', fontsize='x-small')
    plt.ylabel('Number of Students', fontsize='x-small')
    plt.title('Project A', fontsize='x-small')
    plt.xlim(0, 100)
    plt.ylim(0, 30)

    # Show the plot
    plt.tight_layout()
    plt.show()
