#!/usr/bin/env python3
"""This module creates a batch normalization layer for a neural network
in tensorflow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Create a batch normalization layer for a neural network in tensorflow.

    Parameters:
    prev (tf.Tensor): The activated output of the previous layer.
    n (int): The number of nodes in the layer to be created.
    activation (callable): The activation function that should
    be used on the output of the layer.

    Returns:
    tf.Tensor: The activated output for the layer.
    """
    # Create a dense layer
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode="fan_avg"),
        use_bias=False,
    )(prev)

    # Create a batch normalization layer
    norm = tf.keras.layers.BatchNormalization(
        axis=-1,
        epsilon=1e-7,
        beta_initializer=tf.keras.initializers.Zeros(),
        gamma_initializer=tf.keras.initializers.Ones(),
    )(dense)

    # Apply the activation function
    out = activation(norm)

    return out
