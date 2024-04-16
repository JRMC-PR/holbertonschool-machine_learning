#!/usr/bin/env python3
"""
This module contains a function to generate a
histogram representing the distribution of student grades.

The x-axis represents grades from 0 to 100.
The y-axis represents the number of students.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    This function generates a histogram representing
    the distribution of student grades.

    The x-axis represents grades from 0 to 100.
    The y-axis represents the number of students.
    """
    # Set the seed for the random number generator
    # to ensure reproducibility
    np.random.seed(5)
    # Generate a random sample of 50 student grades,
    # normally distributed with mean 68 and standard deviation 15
    student_grades = np.random.normal(68, 15, 50)

    # Create a new figure with a specific size
    plt.figure(figsize=(6.4, 4.8))

    # Label the x-axis as 'Grades'
    plt.xlabel("Grades")
    # Label the y-axis as 'Number of Students'
    plt.ylabel("Number of Students")
    # Set the title of the plot as 'Project A'
    plt.title("Project A")
    # Create a histogram of the student grades, with
    # bins every 10 units from 0 to 100, and black edges on the bars
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    # Add a ticks on the x-axis at every 10 units
    plt.xticks(range(0, 101, 10))
    # Set the x-axis range from 0 to 100
    plt.xlim(0, 100)
    # Set the y-axis range from 0 to 30
    plt.ylim(0, 30)
    # Display the plot
    plt.show()
