#!/usr/bin/env python3
"""This module builds a neural network with keras library"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Build a neural network using Keras.

    Parameters:
    nx (int): Number of input features to the network.
    layers (list): List containing the number of nodes in
    each layer of the network.
    activations (list): List containing the activation
    functions used for each layer of the network.
    lambtha (float): L2 regularization parameter.
    keep_prob (float): Probability that a node will be kept for dropout.

    Returns:
    model: A Keras model instance.
    """

    # Ensure that the number of layers matches the number of activations
    assert len(layers) == len(activations)

    # Initialize a sequential model
    model = Sequential()

    # Add each layer
    for i in range(len(layers)):
        # If it's the first layer, we need to specify the input_dim
        if i == 0:
            model.add(Dense(layers[i], input_dim=nx, activation=activations[i],
                            kernel_regularizer=l2(lambtha)))
        else:
            model.add(Dense(layers[i], activation=activations[i],
                            kernel_regularizer=l2(lambtha)))

        # If it's not the last layer, add dropout
        if i != len(layers) - 1:
            model.add(Dropout(1 - keep_prob))

    return model
