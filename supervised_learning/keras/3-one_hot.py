#!/usr/bin/env python3
"""This module converts a label vestor into a one-hot
matrix"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Convert a label vector into a one-hot matrix.

    Parameters:
    labels (array): The label vector to convert.
    classes (int, optional): The number of classes.
    If not provided, it will be inferred from the labels.

    Returns:
    one_hot_matrix: The one-hot matrix.
    """

    # Use Keras' to_categorical function to convert the
    # labels to a one-hot matrix.
    # If classes is not provided, it will be inferred from
    # the labels.
    one_hot_matrix = \
        K.utils.to_categorical(labels, num_classes=classes)

    return one_hot_matrix
