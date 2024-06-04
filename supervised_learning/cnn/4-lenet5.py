#!/usr/bin/env python3
"""
Module to create a modified LeNet-5 architecture using tensorflow
"""
import tensorflow as tf


def lenet5(x, y):
    """
    Function that builds a modified version of the LeNet-5
    architecture using tensorflow

    Parameters:
    x is a tf.placeholder of shape (m, 28, 28, 1) containing
    the input images for the network
    y is a tf.placeholder of shape (m, 10) containing the
    one-hot labels for the network

    Returns:
    a tensor for the softmax activated output
    a training operation that utilizes Adam optimization
    (with default hyperparameters)
    a tensor for the loss of the netowrk
    a tensor for the accuracy of the network
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=init,
    )(x)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=init,
    )(pool1)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)

    # Flatten the pool2 output
    flat = tf.layers.Flatten()(pool2)

    # Fully connected layer with 120 nodes
    fc1 = \
        tf.layers.Dense(units=120, activation=tf.nn.relu,
                        kernel_initializer=init)(
                        flat
                        )

    # Fully connected layer with 84 nodes
    fc2 = \
        tf.layers.Dense(units=84,
                        activation=tf.nn.relu, kernel_initializer=init)(fc1)

    # Fully connected softmax output layer with 10 nodes
    softmax = tf.layers.Dense(
        units=10, activation=tf.nn.softmax, kernel_initializer=init
    )(fc2)

    # Loss
    loss = tf.losses.softmax_cross_entropy(y, softmax)

    # Training operation
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return softmax, train_op, loss, accuracy
