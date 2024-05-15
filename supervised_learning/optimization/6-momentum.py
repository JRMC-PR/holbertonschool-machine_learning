#!/usr/bin/env python3
"""This moduel contains the function for creating a momentum
optimization operation in Tensorflow
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Set up the gradient descent with momentum
    optimization algorithm in TensorFlow.

    Parameters:
    alpha (float): The learning rate.
    beta1 (float): The momentum weight.

    Returns:
    tf.keras.optimizers.SGD: The SGD optimizer object with momentum.
    """
    # Create a SGD optimizer object with the specified
    # learning rate and momentum weight
    optimizer = \
        tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)

    # Return the optimizer
    return optimizer
