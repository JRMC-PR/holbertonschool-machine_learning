#!/usr/bin/env python3
"""This module saves and loads a model using Keras"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Save an entire model.

    Parameters:
    network (keras model): The model to save.
    filename (str): The path of the file that the
    model should be saved to.

    Returns:
    None
    """
    network.save(filename)


def load_model(filename):
    """
    Load an entire model.

    Parameters:
    filename (str): The path of the file that the
    model should be loaded from.

    Returns:
    The loaded model.
    """
    return K.models.load_model(filename)
