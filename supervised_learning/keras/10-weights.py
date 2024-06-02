#!/usr/bin/env python3
"""This module saves and loads the weights of a
model using Keras"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Save a model's weights.

    Parameters:
    network (keras model): The model whose
    weights should be saved.
    filename (str): The path of the file that
    the weights should be saved to.
    save_format (str): The format in which the
    weights should be saved.

    Returns:
    None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Load a model's weights.

    Parameters:
    network (keras model): The model to which the
    weights should be loaded.
    filename (str): The path of the file that the
    weights should be loaded from.

    Returns:
    None
    """
    network.load_weights(filename)
