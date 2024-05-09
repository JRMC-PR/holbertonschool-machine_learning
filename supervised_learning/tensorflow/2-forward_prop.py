#!/usr/bin/env python3
"""This module difines a function for the
forward prop of a neural network
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow v2 behavior if using TensorFlow v2

# Import the create_layer function
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
    x: Tensor. The placeholder for the input data.
    layer_sizes: List. A list containing the number of nodes in each layer of the network.
    activations: List. A list containing the activation functions for each layer of the network.

    Returns:
    prediction: Tensor. The prediction of the network in tensor form.
    """
    # Initialize the input to the first layer to be x
    layer_input = x

    # Loop over each layer size and corresponding activation function
    for i in range(len(layer_sizes)):
        # If the activation function is None, use a lambda function that returns its input unchanged
        if activations[i] is None:
            act = lambda x: x
        else:
            act = activations[i]

        # Create the layer with the specified size and activation function,
        # using the output of the previous layer as input
        layer_input = create_layer(layer_input, layer_sizes[i], act)

    # The output of the last layer is the prediction of the network
    prediction = layer_input

    return prediction
