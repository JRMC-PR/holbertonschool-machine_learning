#!/usr/bin/env python3
"""This module contains the function for
making predictions"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Make a prediction using a neural network.

    Parameters:
    network (keras model): The network model
    to make the prediction with.
    data (numpy.ndarray): The input data to make
    the prediction with.
    verbose (bool): Determines if output should be
    printed during the prediction process.

    Returns:
    The prediction for the data.
    """
    prediction = network.predict(data, verbose=verbose)
    return prediction
