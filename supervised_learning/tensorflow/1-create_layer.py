#!/usr/bin/env python3
"""This module defines a fucntiopn to create a neuron layer
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Creates a fully connected layer in a neural network
    with He et. al initialization.

    Args:
    prev: Tensor. The output from the previous layer.
    n: Integer. The number of nodes in the layer to create.
    activation: Function. The activation function that the
    layer should use.

    Returns:
    layer: Tensor. The output of the layer after applying weights,
    biases, and activation function.
    """
    # Define the initializer for the weights using
    # He et. al initialization
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Define weights for the current layer
    # Weights are a 2D tensor with shape [prev.shape[1], n],
    # initialized with the defined initializer
    weights = \
        tf.Variable(initializer(shape=(int(prev.shape[1]), n)), name='weights')

    # Define biases for the current layer
    # Biases are a 1D tensor with shape [n], initialized with zeros
    biases = tf.Variable(tf.zeros([n]), name='biases')

    # Compute the weighted sum of the inputs
    # This is done using matrix multiplication between the output
    # of the previous layer and the weights,
    # and then adding the biases
    layer = tf.add(tf.matmul(prev, weights), biases)

    # Apply the activation function to the weighted sum
    layer = activation(layer)

    return layer
