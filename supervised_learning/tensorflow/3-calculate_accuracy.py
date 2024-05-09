#!/usr/bin/env python3
"""This moduel has the method calculate_accuracy"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Args:
    y: Placeholder for the labels of the input data.
    y_pred: Tensor containing the network's predictions.

    Returns:
    A tensor containing the decimal accuracy of the prediction.
    """
    # Use argmax to find the predicted class
    prediction = tf.argmax(y_pred, 1)

    # Use argmax to find the actual class
    actual = tf.argmax(y, 1)

    # Compare the predicted and actual classes
    correct_predictions = tf.equal(prediction, actual)

    # Cast the boolean values to float, and calculate the mean
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy
