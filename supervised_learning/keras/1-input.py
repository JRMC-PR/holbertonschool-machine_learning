#!/usr/bin/env python3
"""This module builds a neural network with Keras library
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with Keras library

    Args:
        nx (int): number of input features to the network
        layers (list): containing the number of nodes in each layer
        of the network
        activations (list): containing the activation functions used
        for each layer of the network
        lambtha (float): L2 regularization parameter
        keep_prob (float): probability that a node will be kept for dropout

    Returns:
        model: Keras model
    """
    # Ensueres that the number of layers matches the number of activations
    assert len(layers) == len(activations)

    # Definethe input layer
    inputs = K.Input(shape=(nx,))

    # Initialize L2 regularization. This is a form of weight decay that encourages
    # the model to have small weights, which helps prevent overfitting.
    regularization = K.regularizers.l2(lambtha)

    # Define the fisrt layer witch is connected to the input layer
    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=regularization)(inputs)

    # Add the rest of the layers
    for i in range(1, len(layers)):
        # Apply dropout before adding the next layer. This randomly
        # sets a fraction '1 - keep_prob'
        # of the input units to 0 at each update during training time,
        # which helps prevent overfitting.
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=regularization)(x)

    # Create the model
    model = K.Model(inputs=inputs, outputs=x)

    return model
