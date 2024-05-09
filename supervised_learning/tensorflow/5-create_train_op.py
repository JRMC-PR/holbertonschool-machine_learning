#!/usr/bin/env python3
"""This module contains the function create_train_op"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.

    Args:
    loss: The loss of the network's prediction.
    alpha: The learning rate.

    Returns:
    An operation that trains the network using gradient descent.
    """
    # Create a GradientDescentOptimizer
    optimizer = tf.train.GradientDescentOptimizer(alpha)

    # Use the optimizer to minimize the loss
    train_op = optimizer.minimize(loss)

    return train_op
