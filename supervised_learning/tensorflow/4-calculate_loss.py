#!/usr/bin/env python3
"""Thismodule contains the function calculate_loss"""

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Args:
    y: Placeholder for the labels of the input data.
    y_pred: Tensor containing the network's predictions.

    Returns:
    A tensor containing the loss of the prediction.
    """
    # Calculate the softmax cross-entropy loss
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    return loss
