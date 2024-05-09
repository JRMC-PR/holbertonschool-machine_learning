#!/usr/bin/env python3
"""This module has the method create_placeholders(nx, classes)"""

import tensorflow as tf
tf.disable_v2_behavior()


def create_pleaceholder(nx, classes):
    """This method create placeholders
    Args:
        nx: the number of feature columns in our data
        classes: the number of classes in our classifier
    Returns:
        x: the placeholder for the input data to the neural network
        y: the placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
